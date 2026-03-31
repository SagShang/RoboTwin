import argparse
from pathlib import Path
import shutil
import subprocess

import cv2
import h5py
import numpy as np


CAMERA_NAMES = ("head_camera", "left_camera", "right_camera")


def decode_bgr_frames(dataset) -> list[np.ndarray]:
    frames = []
    for buf in dataset:
        frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode frame from HDF5 image bytes")
        frames.append(frame)
    return frames


def resize_keep_aspect(image: np.ndarray, target_w: int | None = None, target_h: int | None = None) -> np.ndarray:
    if target_w is None and target_h is None:
        raise ValueError("Either target_w or target_h must be provided")

    src_h, src_w = image.shape[:2]
    if target_w is not None and target_h is not None:
        scale = min(target_w / src_w, target_h / src_h)
    elif target_w is not None:
        scale = target_w / src_w
    else:
        scale = target_h / src_h

    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError(f"No frames to write for {path}")

    h, w = frames[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            for frame in frames:
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            returncode = proc.wait()
        finally:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
            if proc.stderr:
                proc.stderr.close()

        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {path}:\n{stderr}")
        return

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")

    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def build_merged_frames(
    head_frames: list[np.ndarray],
    left_frames: list[np.ndarray],
    right_frames: list[np.ndarray],
) -> list[np.ndarray]:
    frame_count = min(len(head_frames), len(left_frames), len(right_frames))
    if frame_count == 0:
        return []

    left_w = left_frames[0].shape[1]
    right_w = right_frames[0].shape[1]
    canvas_w = left_w + right_w

    merged_frames = []
    for idx in range(frame_count):
        top = resize_keep_aspect(head_frames[idx], target_w=canvas_w)
        bottom_left = resize_keep_aspect(left_frames[idx], target_w=left_w)
        bottom_right = resize_keep_aspect(right_frames[idx], target_w=right_w)
        bottom = np.concatenate([bottom_left, bottom_right], axis=1)
        merged_frames.append(np.concatenate([top, bottom], axis=0))

    return merged_frames


def infer_fps(dataset_root: Path, episode_stem: str, default_fps: float) -> float:
    video_path = dataset_root / "video" / f"{episode_stem}.mp4"
    if not video_path.exists():
        return default_fps

    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()

    return fps if fps and fps > 0 else default_fps


def export_episode(hdf5_path: Path, output_dir: Path, fps: float) -> None:
    with h5py.File(hdf5_path, "r") as f:
        frames = {cam: decode_bgr_frames(f[f"observation/{cam}/rgb"]) for cam in CAMERA_NAMES}

    output_dir.mkdir(parents=True, exist_ok=True)

    merged_frames = build_merged_frames(
        frames["head_camera"],
        frames["left_camera"],
        frames["right_camera"],
    )
    write_video(output_dir / f"{hdf5_path.stem}_three_view.mp4", merged_frames, fps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RobotWin three-view videos from HDF5")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Dataset root such as data/stack_blocks_two/franka_clean_50",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="Fallback FPS when no reference video exists",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="video/three_view_exports",
        help="Output path. Relative paths are resolved from dataset_root",
    )
    args = parser.parse_args()

    dataset_root = args.input.resolve()
    data_dir = dataset_root / "data"
    output_path = Path(args.output)
    output_dir = output_path if output_path.is_absolute() else (dataset_root / output_path)

    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found under {data_dir}")

    for hdf5_path in hdf5_files:
        fps = infer_fps(dataset_root, hdf5_path.stem, args.fps)
        export_episode(hdf5_path, output_dir, fps)
        print(f"Exported {hdf5_path.name} to {output_dir} at {fps:.2f} FPS")


if __name__ == "__main__":
    main()

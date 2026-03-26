import base64
import json
import socket
from typing import Any

import numpy as np


def encode_obs(observation, prompt: str):
    return {
        "images": {
            "cam_high": np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0),
            "cam_left_wrist": np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0),
            "cam_right_wrist": np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0),
        },
        "state": observation["joint_action"]["vector"],
        "prompt": prompt,
    }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def numpy_to_json(data: Any) -> str:
    return json.dumps(data, cls=NumpyEncoder)


def json_to_numpy(json_str: str) -> Any:
    def object_hook(dct: dict[str, Any]):
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    return json.loads(json_str, object_hook=object_hook)


class RemoteOpenPIClient:
    def __init__(self, host="127.0.0.1", port=None, timeout=30, open_loop_steps=50):
        if port is None:
            raise ValueError("server_port is required for openpi client")

        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self.open_loop_steps = int(open_loop_steps)
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)

    def _recv_exact(self, size):
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = self.sock.recv(min(remaining, 4096))
            if not chunk:
                raise ConnectionError("Connection closed by remote openpi server")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _request(self, cmd, obs=None):
        payload = numpy_to_json({"cmd": cmd, "obs": obs}).encode("utf-8")
        self.sock.sendall(len(payload).to_bytes(4, "big"))
        self.sock.sendall(payload)

        response_size = int.from_bytes(self._recv_exact(4), "big")
        response = json_to_numpy(self._recv_exact(response_size).decode("utf-8"))
        if not response.get("ok", False):
            error = response.get("error", "Unknown remote server error")
            trace = response.get("traceback", "")
            raise RuntimeError(f"{error}\n{trace}".strip())
        return response.get("result")

    def get_action(self, obs):
        return self._request("get_action", obs)

    def reset_model(self):
        self._request("reset_model")

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None


def get_model(usr_args):
    return RemoteOpenPIClient(
        host=usr_args.get("server_host", "127.0.0.1"),
        port=usr_args.get("server_port"),
        timeout=usr_args.get("timeout", 30),
        open_loop_steps=usr_args.get("open_loop_steps", usr_args.get("pi0_step", 50)),
    )


def eval(TASK_ENV, model, observation):
    prompt = TASK_ENV.get_instruction()
    obs = encode_obs(observation, prompt)
    result = model.get_action(obs)
    actions = result["actions"][: model.open_loop_steps]

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()


def reset_model(model):
    model.reset_model()

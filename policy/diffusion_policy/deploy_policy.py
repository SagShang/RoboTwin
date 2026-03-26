import base64
import json
import socket
from typing import Any

import numpy as np


def encode_obs(observation):
    head_cam = np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255
    left_cam = np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255
    right_cam = np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255
    obs = {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "agent_pos": observation["joint_action"]["vector"],
    }
    return obs


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
    def object_hook(dct):
        if "__numpy_array__" in dct:
            raw = base64.b64decode(dct["data"])
            return np.frombuffer(raw, dtype=dct["dtype"]).reshape(dct["shape"])
        return dct

    return json.loads(json_str, object_hook=object_hook)


class RemoteDPClient:
    def __init__(self, host="127.0.0.1", port=None, timeout=30):
        if port is None:
            raise ValueError("server_port is required for diffusion_policy client")

        self.host = host
        self.port = int(port)
        self.timeout = timeout
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)

    def _recv_exact(self, size):
        chunks = []
        remaining = size
        while remaining > 0:
            chunk = self.sock.recv(min(remaining, 4096))
            if not chunk:
                raise ConnectionError("Connection closed by remote diffusion policy server")
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
            traceback = response.get("traceback", "")
            raise RuntimeError(f"{error}\n{traceback}".strip())
        return response.get("result")

    def get_action(self, obs):
        return self._request("get_action", obs)

    def update_obs(self, obs):
        self._request("update_obs", obs)

    def reset_obs(self):
        self._request("reset_obs")

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None


def get_model(usr_args):
    return RemoteDPClient(
        host=usr_args.get("server_host", "127.0.0.1"),
        port=usr_args.get("server_port"),
        timeout=usr_args.get("timeout", 30),
    )


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)
    actions = model.get_action(obs)

    for action in actions:
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)


def reset_model(model):
    model.reset_obs()

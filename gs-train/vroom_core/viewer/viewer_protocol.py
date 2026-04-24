"""Minimal network GUI bridge for interactive VRoom checkpoint viewing."""

from __future__ import annotations

import json
import math
import socket
import traceback

import torch


host = "127.0.0.1"
port = 6009
conn = None
addr = None
listener = None


class GuiCamera:
    """Camera object shaped for VRoom rasterization from GUI messages."""

    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.inverse(self.world_view_transform)[3][:3]
        self.fx = self.image_width / (2.0 * math.tan(self.FoVx / 2.0))
        self.fy = self.image_height / (2.0 * math.tan(self.FoVy / 2.0))
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0


def init(wish_host: str, wish_port: int):
    global host, port, listener
    host = wish_host
    port = wish_port
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind((host, port))
    listener.listen()
    listener.settimeout(0)


def try_connect():
    global conn, addr
    if listener is None:
        return
    try:
        conn, addr = listener.accept()
        print(f"\nConnected by {addr}")
        conn.settimeout(None)
    except Exception:
        pass


def _read_message():
    message_length = conn.recv(4)
    message_length = int.from_bytes(message_length, "little")
    message = conn.recv(message_length)
    return json.loads(message.decode("utf-8"))


def send(message_bytes, verify):
    if message_bytes is not None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, "little"))
    conn.sendall(bytes(verify, "ascii"))


def receive():
    message = _read_message()
    width = message["resolution_x"]
    height = message["resolution_y"]
    if width == 0 or height == 0:
        return None, None, None, None

    try:
        do_training = bool(message["train"])
        keep_alive = bool(message["keep_alive"])
        add_prefilter = bool(message["rot_scale_python"])
        fovy = message["fov_y"]
        fovx = message["fov_x"]
        znear = message["z_near"]
        zfar = message["z_far"]
        world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
        world_view_transform[:, 1] = -world_view_transform[:, 1]
        world_view_transform[:, 2] = -world_view_transform[:, 2]
        full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
        full_proj_transform[:, 1] = -full_proj_transform[:, 1]
        custom_cam = GuiCamera(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
    except Exception as exc:
        print("")
        traceback.print_exc()
        raise exc

    return custom_cam, do_training, add_prefilter, keep_alive

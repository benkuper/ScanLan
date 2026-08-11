from __future__ import annotations

import collections
import io
import queue
import struct
import threading
import time
from dataclasses import dataclass, replace
from typing import BinaryIO

import numpy as np


RGBD_MAGIC = b"SCANRGBD"
RGBD_VERSION = 1
RGBD_HAS_COLOR = 1 << 0
RGBD_HAS_IMU_DELTA = 1 << 1
RGBD_HAS_CAMERA_POSE = 1 << 2
RGBD_MIRROR_X = 1 << 3
RGBD_HEADER = struct.Struct("<8sHHIQQQII7f4f16fII")
MAX_FRAME_PIXELS = 4096 * 4096


@dataclass(frozen=True)
class StreamCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale: float
    min_depth_m: float
    max_depth_m: float


@dataclass(frozen=True)
class RgbdFrame:
    sequence: int
    depth_timestamp_us: int
    color_timestamp_us: int
    camera: StreamCamera
    depth: np.ndarray
    color: np.ndarray | None
    gyro_delta_xyzw: np.ndarray | None
    camera_to_world: np.ndarray | None
    mirror_x: bool = False


class RgbdStreamError(RuntimeError):
    pass


def _read_exact(source: BinaryIO, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = source.read(remaining)
        if not chunk:
            if remaining == count:
                raise EOFError
            raise RgbdStreamError(
                f"RGB-D stream ended with {remaining} of {count} bytes missing"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_rgbd_frame(source: BinaryIO) -> RgbdFrame:
    header_bytes = _read_exact(source, RGBD_HEADER.size)
    values = RGBD_HEADER.unpack(header_bytes)
    (
        magic,
        version,
        encoded_header_size,
        flags,
        sequence,
        depth_timestamp_us,
        color_timestamp_us,
        width,
        height,
        fx,
        fy,
        cx,
        cy,
        depth_scale,
        min_depth_m,
        max_depth_m,
        gyro_x,
        gyro_y,
        gyro_z,
        gyro_w,
        *tail,
    ) = values
    camera_to_world_values = tail[:16]
    depth_bytes, color_bytes = tail[16:]

    if magic != RGBD_MAGIC:
        raise RgbdStreamError(f"Unknown RGB-D stream magic {magic!r}")
    if version != RGBD_VERSION or encoded_header_size != RGBD_HEADER.size:
        raise RgbdStreamError(
            f"Unsupported RGB-D stream header version={version} size={encoded_header_size}"
        )
    pixel_count = int(width) * int(height)
    if width <= 0 or height <= 0 or pixel_count > MAX_FRAME_PIXELS:
        raise RgbdStreamError(f"Invalid RGB-D frame dimensions {width}x{height}")
    if not np.isfinite([fx, fy, cx, cy, depth_scale, min_depth_m, max_depth_m]).all():
        raise RgbdStreamError("RGB-D frame calibration contains a non-finite value")
    if fx <= 0 or fy <= 0 or depth_scale <= 0 or max_depth_m <= min_depth_m:
        raise RgbdStreamError("RGB-D frame calibration is physically invalid")
    expected_depth_bytes = pixel_count * 2
    expected_color_bytes = pixel_count * 3 if flags & RGBD_HAS_COLOR else 0
    if depth_bytes != expected_depth_bytes or color_bytes != expected_color_bytes:
        raise RgbdStreamError(
            "RGB-D payload size does not match its dimensions "
            f"(depth {depth_bytes}/{expected_depth_bytes}, color {color_bytes}/{expected_color_bytes})"
        )

    depth_payload = _read_exact(source, depth_bytes)
    color_payload = _read_exact(source, color_bytes) if color_bytes else None
    depth = np.frombuffer(depth_payload, dtype="<u2").reshape(height, width).copy()
    color = (
        np.frombuffer(color_payload, dtype=np.uint8).reshape(height, width, 3).copy()
        if color_payload is not None
        else None
    )
    gyro_delta = (
        np.asarray([gyro_x, gyro_y, gyro_z, gyro_w], dtype=np.float64)
        if flags & RGBD_HAS_IMU_DELTA
        else None
    )
    camera_to_world = (
        np.asarray(camera_to_world_values, dtype=np.float64).reshape(4, 4)
        if flags & RGBD_HAS_CAMERA_POSE
        else None
    )
    return RgbdFrame(
        sequence=int(sequence),
        depth_timestamp_us=int(depth_timestamp_us),
        color_timestamp_us=int(color_timestamp_us),
        camera=StreamCamera(
            width=int(width),
            height=int(height),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            depth_scale=float(depth_scale),
            min_depth_m=float(min_depth_m),
            max_depth_m=float(max_depth_m),
        ),
        depth=depth,
        color=color,
        gyro_delta_xyzw=gyro_delta,
        camera_to_world=camera_to_world,
        mirror_x=bool(flags & RGBD_MIRROR_X),
    )


def reject_depth_speckles(depth: np.ndarray, camera: StreamCamera) -> np.ndarray:
    """Remove unsupported one-pixel returns with an edge-aware neighbour test."""

    values = np.asarray(depth, dtype=np.uint16)
    if values.shape != (camera.height, camera.width):
        raise ValueError("Depth array does not match the stream camera")
    if camera.width < 3 or camera.height < 3:
        return values.copy()
    valid = values > 0
    support = np.zeros(values.shape, dtype=np.uint8)
    millimetres = values.astype(np.int32, copy=False)

    def compare(
        center: tuple[slice, slice],
        neighbour: tuple[slice, slice],
    ) -> None:
        center_depth = millimetres[center]
        neighbour_depth = millimetres[neighbour]
        threshold = np.maximum(24, np.rint(center_depth * 0.015).astype(np.int32))
        support[center] += (
            valid[center]
            & valid[neighbour]
            & (np.abs(center_depth - neighbour_depth) <= threshold)
        )

    compare((slice(None), slice(1, None)), (slice(None), slice(None, -1)))
    compare((slice(None), slice(None, -1)), (slice(None), slice(1, None)))
    compare((slice(1, None), slice(None)), (slice(None, -1), slice(None)))
    compare((slice(None, -1), slice(None)), (slice(1, None), slice(None)))
    cleaned = values.copy()
    cleaned[valid & (support == 0)] = 0
    return cleaned


def encode_rgbd_frame(frame: RgbdFrame) -> bytes:
    camera = frame.camera
    depth = np.asarray(frame.depth, dtype="<u2")
    if depth.shape != (camera.height, camera.width):
        raise ValueError("Depth array does not match the stream camera")
    color_bytes = b""
    flags = 0
    if frame.color is not None:
        color = np.asarray(frame.color, dtype=np.uint8)
        if color.shape != (camera.height, camera.width, 3):
            raise ValueError("Color array does not match the stream camera")
        color_bytes = color.tobytes()
        flags |= RGBD_HAS_COLOR
    gyro = np.asarray(
        frame.gyro_delta_xyzw if frame.gyro_delta_xyzw is not None else [0, 0, 0, 1],
        dtype=np.float32,
    )
    if gyro.shape != (4,):
        raise ValueError("Gyro delta must be an XYZW quaternion")
    if frame.gyro_delta_xyzw is not None:
        flags |= RGBD_HAS_IMU_DELTA
    pose = np.asarray(
        frame.camera_to_world if frame.camera_to_world is not None else np.eye(4),
        dtype=np.float32,
    )
    if pose.shape != (4, 4):
        raise ValueError("Camera pose must be 4x4")
    if frame.camera_to_world is not None:
        flags |= RGBD_HAS_CAMERA_POSE
    if frame.mirror_x:
        flags |= RGBD_MIRROR_X
    depth_bytes = depth.tobytes()
    header = RGBD_HEADER.pack(
        RGBD_MAGIC,
        RGBD_VERSION,
        RGBD_HEADER.size,
        flags,
        frame.sequence,
        frame.depth_timestamp_us,
        frame.color_timestamp_us,
        camera.width,
        camera.height,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.depth_scale,
        camera.min_depth_m,
        camera.max_depth_m,
        *gyro,
        *pose.reshape(-1),
        len(depth_bytes),
        len(color_bytes),
    )
    return header + depth_bytes + color_bytes


class LatestFrameQueue:
    """A bounded latest-wins queue with observable drop accounting."""

    def __init__(self, capacity: int = 4) -> None:
        if capacity <= 0:
            raise ValueError("Frame queue capacity must be positive")
        self._capacity = capacity
        self._frames: collections.deque[RgbdFrame] = collections.deque()
        self._available = threading.Condition()
        self.dropped = 0

    def put(self, frame: RgbdFrame) -> None:
        with self._available:
            if len(self._frames) == self._capacity:
                dropped = self._frames.popleft()
                self.dropped += 1
                if self._frames:
                    self._frames[0] = _prepend_gyro_delta(self._frames[0], dropped)
                else:
                    frame = _prepend_gyro_delta(frame, dropped)
            self._frames.append(frame)
            self._available.notify()

    def get(self, timeout: float | None = None) -> RgbdFrame:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._available:
            while not self._frames:
                if deadline is None:
                    self._available.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise queue.Empty
                self._available.wait(remaining)
            return self._frames.popleft()

    def qsize(self) -> int:
        with self._available:
            return len(self._frames)


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = np.asarray(left, dtype=np.float64)
    rx, ry, rz, rw = np.asarray(right, dtype=np.float64)
    value = np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(value))
    if np.isfinite(norm) and norm > 1e-12:
        return value / norm
    return np.asarray([0, 0, 0, 1])


def _prepend_gyro_delta(target: RgbdFrame, dropped: RgbdFrame) -> RgbdFrame:
    """Preserve the complete rotation interval when a queued image is dropped."""

    earlier = dropped.gyro_delta_xyzw
    if earlier is None:
        return target
    later = target.gyro_delta_xyzw
    combined = (
        np.asarray(earlier, dtype=np.float64).copy()
        if later is None
        else _quaternion_multiply_xyzw(later, earlier)
    )
    return replace(target, gyro_delta_xyzw=combined)


def decode_rgbd_frame(payload: bytes) -> RgbdFrame:
    source = io.BytesIO(payload)
    frame = read_rgbd_frame(source)
    if source.read(1):
        raise RgbdStreamError("RGB-D frame has trailing bytes")
    return frame

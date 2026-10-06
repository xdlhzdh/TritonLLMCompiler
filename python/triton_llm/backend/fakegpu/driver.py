"""Driver for the teaching backend.

``is_active`` is false, so Triton keeps ``CudaDriver`` until ``activate``
calls ``driver.set_active``. Device id 99 keeps this JIT cache away from
CUDA device 0.
"""
from __future__ import annotations

from triton.backends.compiler import GPUTarget
from triton.backends.driver import DriverBase

FAKE_DEVICE = 99
LAUNCHES: list[tuple] = []


class FakeUtils:
    def get_device_properties(self, device):
        return {"max_shared_mem": 96 * 1024}

    def load_binary(self, name, kernel, shared, device):
        return ("fakegpu-module", kernel, 0, 0)


class FakeLauncher:
    def __init__(self, src, metadata):
        self.name = metadata.name

    def __call__(self, grid_x, grid_y, grid_z, stream, function, packed_metadata, *args):
        LAUNCHES.append((self.name, (grid_x, grid_y, grid_z), stream, packed_metadata))


class FakeGPUDriver(DriverBase):
    def __init__(self) -> None:
        self.utils = FakeUtils()
        self.launcher_cls = FakeLauncher

    @staticmethod
    def is_active():
        return False

    def get_current_device(self):
        return FAKE_DEVICE

    def get_current_stream(self, device):
        return 0

    def get_current_target(self):
        return GPUTarget("fakegpu", 0, 32)

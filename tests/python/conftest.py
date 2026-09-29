import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires CUDA device")


@pytest.fixture(scope="session")
def cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")

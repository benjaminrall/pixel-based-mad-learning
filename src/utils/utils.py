import torch
from torch.types import Device
import random
import numpy as np
from typing import Any

DEVICE: Device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_device() -> Device:
    """Returns the device available to be used by torch."""
    return DEVICE

def to_nested_dict(obj: Any) -> dict[str, Any]:
    """
    Recursively converts an object and its nested objects into a dictionary.
    Used to upload Trainer configs to W&B.
    """
    if isinstance(obj, dict):
        return {str(k): to_nested_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {
            str(k): to_nested_dict(v) for k, v in vars(obj).items()
            if not str(k).startswith("_") and not callable(v)
        }
    elif isinstance(obj, (list, tuple)):
        return {str(i): to_nested_dict(v) for i, v in enumerate(obj)}
    else:
        return obj

def get_random_state() -> dict:
    """Returns a dictionary containing the current random state."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())]
    }

def restore_random_state(state: dict) -> None:
    """Restores a random state from a dictionary."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    for i, cuda_state in enumerate(state["cuda"]):
        torch.cuda.set_rng_state(cuda_state, device=i)

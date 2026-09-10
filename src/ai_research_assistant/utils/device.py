import torch


def resolve_device(device: str) -> str:
    
    """Resolves 'auto' to the best available device; an explicit choice
    (cpu / cuda / cuda:0 / mps) passes through unchanged.
    """
    
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"

    return "cpu"
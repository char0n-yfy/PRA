import os

import torch

from utils.logging_util import log_for_0


def restore_checkpoint(state, workdir):
    """
    Restore a checkpoint saved by ``save_checkpoint`` into a train-state-like object.
    """
    path = workdir
    if os.path.isdir(path):
        path = os.path.join(path, "checkpoint_last.pt")
    if not os.path.exists(path):
        log_for_0(f"No checkpoint found at {path}.")
        return state

    ckpt = torch.load(path, map_location="cpu")
    if hasattr(state, "load_state_dict"):
        state.load_state_dict(ckpt)
    else:
        raise TypeError("state must implement load_state_dict for restore_checkpoint.")
    log_for_0(f"Restored from checkpoint at {path}")
    return state


def save_checkpoint(state, workdir):
    """
    Save a train-state-like object to ``workdir/checkpoint_last.pt``.
    """
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, "checkpoint_last.pt")
    if hasattr(state, "state_dict"):
        ckpt = state.state_dict()
    else:
        raise TypeError("state must implement state_dict for save_checkpoint.")
    torch.save(ckpt, path)
    step = int(getattr(state, "step", 0))
    log_for_0(f"Checkpoint step {step} saved at {path}.")

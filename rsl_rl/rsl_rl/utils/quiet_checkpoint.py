"""Audited baseline initialization and strict quiet-experiment restoration."""

import copy
import hashlib
import os

import torch

from rsl_rl.modules.quiet_actor_critic import QuietHIMActorCritic


QUIET_CHECKPOINT_VERSION = 1


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_base_checkpoint(model, path, map_location='cpu'):
    """Initialize only model tensors; keep the source checkpoint read-only."""
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
        raise ValueError("Baseline checkpoint must contain model_state_dict.")
    model.load_base_model(checkpoint['model_state_dict'])
    return {
        'base_checkpoint': os.path.abspath(path),
        'base_checkpoint_sha256': file_sha256(path),
        'base_iteration': int(checkpoint.get('iter', 0)),
        'format_version': QUIET_CHECKPOINT_VERSION,
        'policy': copy.deepcopy(model.policy_spec),
    }


def build_quiet_checkpoint(model, optimizer, iteration, metadata=None, infos=None):
    """Build the normal runner payload with explicit experiment metadata."""
    quiet_metadata = copy.deepcopy(metadata or {})
    quiet_metadata['format_version'] = QUIET_CHECKPOINT_VERSION
    quiet_metadata['policy'] = copy.deepcopy(model.policy_spec)
    return {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'iter': int(iteration),
        'quiet_metadata': quiet_metadata,
        'infos': infos,
    }


def save_quiet_checkpoint(model, path, optimizer=None, iteration=0, metadata=None, infos=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    checkpoint = build_quiet_checkpoint(model, optimizer, iteration, metadata, infos)
    torch.save(checkpoint, path)
    return checkpoint


def validate_quiet_checkpoint(model, checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Quiet checkpoint must be a dictionary.")
    required = {'model_state_dict', 'optimizer_state_dict', 'iter', 'quiet_metadata'}
    if not required.issubset(checkpoint):
        raise ValueError("Quiet checkpoint is missing fields: {}".format(
            sorted(required - set(checkpoint))))
    metadata = checkpoint['quiet_metadata']
    if metadata.get('format_version') != QUIET_CHECKPOINT_VERSION:
        raise ValueError("Unsupported quiet checkpoint format.")
    if metadata.get('policy') != model.policy_spec:
        raise ValueError("Quiet checkpoint policy configuration differs from this model.")
    if int(checkpoint['iter']) < 0:
        raise ValueError("Quiet checkpoint iteration must be nonnegative.")
    current = model.state_dict()
    saved = checkpoint['model_state_dict']
    missing, unexpected = set(current) - set(saved), set(saved) - set(current)
    mismatched = [name for name in set(current).intersection(saved)
                  if current[name].shape != saved[name].shape]
    if missing or unexpected or mismatched:
        raise RuntimeError("Invalid quiet model state: missing={}, unexpected={}, "
                           "shape_mismatch={}".format(sorted(missing),
                                                     sorted(unexpected),
                                                     sorted(mismatched)))
    if any(not torch.isfinite(tensor).all().item() for tensor in saved.values()
           if tensor.is_floating_point()):
        raise ValueError("Quiet checkpoint contains non-finite model tensors.")
    for name in ('std', 'base_std'):
        if torch.any(saved[name] <= 0).item():
            raise ValueError("Action standard deviation must be positive.")


def load_quiet_checkpoint(model, path, optimizer=None, map_location='cpu'):
    checkpoint = torch.load(path, map_location=map_location)
    validate_quiet_checkpoint(model, checkpoint)
    if optimizer is not None and checkpoint['optimizer_state_dict'] is None:
        raise ValueError("Resume requires saved optimizer state.")
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.set_adapter_enabled(True)
    return checkpoint


def load_quiet_policy(path, device='cpu', adapter_enabled=True):
    """Recreate a policy for evaluation/export from its saved architecture."""
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or 'quiet_metadata' not in checkpoint:
        raise ValueError("Expected a quiet experiment checkpoint.")
    metadata = checkpoint['quiet_metadata']
    if 'policy' not in metadata:
        raise ValueError("Quiet checkpoint has no policy configuration.")
    model = QuietHIMActorCritic(**metadata['policy']).to(device)
    validate_quiet_checkpoint(model, checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.set_adapter_enabled(adapter_enabled)
    model.eval()
    return model, checkpoint

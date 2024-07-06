import math
from typing import Mapping


def remove_common_prefix(
        state_dict: Mapping[str, object],
        prefixes: list[str],
) -> Mapping[str, object]:
    if len(state_dict) > 0:
        for prefix in prefixes:
            if all(i.startswith(prefix) for i in state_dict.keys()):
                state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def canonicalize_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """
    Canonicalize a state dict.

    This function is used to canonicalize a state dict, so that it can be
    used for architecture detection and loading.

    This function is not intended to be used in production code.
    """

    # the real state dict might be inside a dict with a known key
    unwrap_keys = ["state_dict", "params_ema", "params-ema", "params", "model", "net"]
    for unwrap_key in unwrap_keys:
        if unwrap_key in state_dict and isinstance(state_dict[unwrap_key], dict):
            state_dict = state_dict[unwrap_key]
            break

    # remove known common prefixes
    state_dict = remove_common_prefix(state_dict, ["module.", "netG."])

    return state_dict


def pixelshuffle_scale(ps_size: int, channels: int):
    return int(math.sqrt(ps_size / channels))


def dysample_scale(ds_size: int):
    return int(math.sqrt(ds_size / 8))


def get_seq_len(state_dict: Mapping[str, object], seq_key: str) -> int:
    """
    Returns the length of a sequence in the state dict.

    The length is detected by finding the maximum index `i` such that
    `{seq_key}.{i}.{suffix}` is in `state` for some suffix.

    Example:
        get_seq_len(state, "body") -> 5
    """
    prefix = seq_key + "."

    keys: set[int] = set()
    for k in state_dict.keys():
        if k.startswith(prefix):
            index = k[len(prefix):].split(".", maxsplit=1)[0]
            keys.add(int(index))

    if len(keys) == 0:
        return 0
    return max(keys) + 1

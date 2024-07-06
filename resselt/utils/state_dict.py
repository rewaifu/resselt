import math
from typing import Mapping


def remove_common_prefix(
    state_dict: Mapping[str, object],
    prefixes: list[str],
) -> Mapping[str, object]:
    if len(state_dict) > 0:
        for prefix in prefixes:
            if all(i.startswith(prefix) for i in state_dict.keys()):
                state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def canonicalize_state_dict(state_dict: Mapping[str, object]) -> Mapping[str, object]:
    """
    Canonicalize a state dict.

    This function is used to canonicalize a state dict, so that it can be
    used for architecture detection and loading.

    This function is not intended to be used in production code.
    """

    # the real state dict might be inside a dict with a known key
    unwrap_keys = ['state_dict', 'params_ema', 'params-ema', 'params', 'model', 'net']
    for unwrap_key in unwrap_keys:
        if unwrap_key in state_dict and isinstance(state_dict[unwrap_key], dict):
            state_dict = state_dict[unwrap_key]
            break

    # remove known common prefixes
    state_dict = remove_common_prefix(state_dict, ['module.', 'netG.'])

    return state_dict


def pixelshuffle_scale(ps_size: int, channels: int):
    return math.isqrt(ps_size // channels)


def dysample_scale(ds_size: int):
    return math.isqrt(ds_size // 8)


def get_pixelshuffle_params(
    state_dict: Mapping[str, object],
    upsample_key: str = 'upsample',
    default_nf: int = 64,
) -> tuple[int, int]:
    """
    This will detect the upscale factor and number of features of a pixelshuffle module in the state dict.

    A pixelshuffle module is a sequence of alternating up convolutions and pixelshuffle.
    The class of this module is commonyl called `Upsample`.
    Examples of such modules can be found in most SISR architectures, such as SwinIR, HAT, RGT, and many more.
    """
    upscale = 1
    num_feat = default_nf

    for i in range(0, 10, 2):
        key = f'{upsample_key}.{i}.weight'
        if key not in state_dict:
            break

        tensor = state_dict[key]
        # we'll assume that the state dict contains tensors
        shape: tuple[int, ...] = tensor.shape  # type: ignore
        num_feat = shape[1]
        upscale *= math.isqrt(shape[0] // num_feat)

    return upscale, num_feat


def get_seq_len(state_dict: Mapping[str, object], seq_key: str) -> int:
    """
    Returns the length of a sequence in the state dict.

    The length is detected by finding the maximum index `i` such that
    `{seq_key}.{i}.{suffix}` is in `state` for some suffix.

    Example:
        get_seq_len(state, "body") -> 5
    """
    prefix = seq_key + '.'

    keys: set[int] = set()
    for k in state_dict.keys():
        if k.startswith(prefix):
            index = k[len(prefix) :].split('.', maxsplit=1)[0]
            keys.add(int(index))

    if len(keys) == 0:
        return 0
    return max(keys) + 1

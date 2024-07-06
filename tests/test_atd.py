from resselt.archs.atd import ATD, ATDArch
from tests.utils import ModelAsset, assert_loads_correctly, assert_image_inference, disallowed_props, ImageAssets


def test_atd_load():
    assert_loads_correctly(
        ATDArch(),
        lambda: ATD(),
        lambda: ATD(in_chans=4, embed_dim=60),
        lambda: ATD(window_size=4),
        lambda: ATD(depths=(4, 6, 8, 7, 5), num_heads=(4, 6, 8, 12, 5)),
        lambda: ATD(num_tokens=32, reducted_dim=3, convffn_kernel_size=7, mlp_ratio=3),
        lambda: ATD(qkv_bias=False),
        lambda: ATD(patch_norm=False),
        lambda: ATD(ape=True),
        lambda: ATD(resi_connection='1conv'),
        lambda: ATD(resi_connection='3conv'),
        lambda: ATD(upsampler='', upscale=1),
        lambda: ATD(upsampler='nearest+conv', upscale=4),
        lambda: ATD(upsampler='pixelshuffle', upscale=1),
        lambda: ATD(upsampler='pixelshuffle', upscale=2),
        lambda: ATD(upsampler='pixelshuffle', upscale=3),
        lambda: ATD(upsampler='pixelshuffle', upscale=4),
        lambda: ATD(upsampler='pixelshuffle', upscale=8),
        lambda: ATD(upsampler='pixelshuffledirect', upscale=1),
        lambda: ATD(upsampler='pixelshuffledirect', upscale=2),
        lambda: ATD(upsampler='pixelshuffledirect', upscale=3),
        lambda: ATD(upsampler='pixelshuffledirect', upscale=4),
        lambda: ATD(upsampler='pixelshuffledirect', upscale=8),
    )


def test_atd_inference(snapshot):
    asset = ModelAsset(filename='4x_DWTP_DS_ATDl.pth', url='https://public.yor.ovh/pytorch_models/4x_DWTP_DS_ATDl.pth')
    wrapped_model = asset.load_wrapped_model(ATDArch())

    assert wrapped_model == snapshot(exclude=disallowed_props)
    assert isinstance(wrapped_model.model, ATD)

    assert_image_inference(wrapped_model, asset.filename, ImageAssets.COLOR_CAT_120_113)

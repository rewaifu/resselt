from resselt.archs.spanplus import SpanPlus, SpanPlusArch
from tests.utils import ModelAsset, assert_loads_correctly, assert_image_inference, disallowed_props, ImageAssets


def test_spanplus_load():
    assert_loads_correctly(
        SpanPlusArch(),
        lambda: SpanPlus(num_in_ch=3, num_out_ch=3),
        lambda: SpanPlus(num_in_ch=1, num_out_ch=3),
        lambda: SpanPlus(num_in_ch=1, num_out_ch=1),
        lambda: SpanPlus(num_in_ch=4, num_out_ch=4),
        lambda: SpanPlus(num_in_ch=3, num_out_ch=3, feature_channels=32),
        lambda: SpanPlus(num_in_ch=3, num_out_ch=3, feature_channels=64),
        lambda: SpanPlus(num_in_ch=3, num_out_ch=3, upscale=1),
        lambda: SpanPlus(num_in_ch=3, num_out_ch=3, upscale=2),
        lambda: SpanPlus(num_in_ch=3, num_out_ch=3, upscale=4),
        lambda: SpanPlus(num_in_ch=3, num_out_ch=3, upscale=8),
        lambda: SpanPlus(num_in_ch=3, num_out_ch=3, blocks=[2]),
        lambda: SpanPlus(num_in_ch=3, num_out_ch=3, blocks=[4, 4, 4]),
        lambda: SpanPlus(num_in_ch=3, num_out_ch=3, upsampler='ps'),
    )


def test_spanplus_inference(snapshot):
    asset = ModelAsset(filename='2x_spanplus.pth', url='https://public.yor.ovh/torch_models/2x_spanplus.pth')
    wrapped_model = asset.load_wrapped_model(SpanPlusArch())

    assert wrapped_model == snapshot(exclude=disallowed_props)
    assert isinstance(wrapped_model.model, SpanPlus)

    assert_image_inference(wrapped_model, asset.filename, ImageAssets.COLOR_120_113)

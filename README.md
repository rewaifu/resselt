# [~~nu~~]resselt

# Usage

```py
import torch
from pepeline import read, save

from resselt import global_registry
from resselt.utils import upscale_with_tiler, AutoTiler

device = torch.device("cuda")
torch.set_default_device(device)

state_dict = torch.load("spanplus_2x.pth")
wrapped_model = global_registry.load_from_state_dict(state_dict)

img = read("test.jpg", None, 0)
tiler = AutoTiler(wrapped_model)

output_img = upscale_with_tiler(img, tiler, wrapped_model, device)
save(output_img, "output.png")
```

## Build
First, download the pixi tool from https://pixi.sh/latest/ and run the following command:
```bash
> pixi install -e build
> pixi run build
```


## Supported architectures
* [ATD](https://github.com/LabShuHangGU/Adaptive-Token-Dictionary)
* [SRVGGNetCompact](https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/srvgg_arch.py)
* [Real-Cugan](https://github.com/bilibili/ailab)
* [DAT](https://github.com/zhengchen1999/dat)
* [Esrgan](https://github.com/xinntao/Real-ESRGAN)
* [OmniSR](https://github.com/Francis0625/Omni-SR)
* [PLKSR](https://github.com/dslisleedh/PLKSR)
* [RGT](https://github.com/zhengchen1999/RGT)
* [SPAN](https://github.com/hongyuanyu/span)
* [SPANPlus](https://github.com/umzi2/spanplus)
* [SwinIR](https://github.com/JingyunLiang/SwinIR)

## Credits
* Based on [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel)
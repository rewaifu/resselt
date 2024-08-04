# [~~nu~~]resselt

# Usage

```py
import torch
from pepeline import read, save

from resselt import global_registry
from resselt.utils import upscale_with_tiler, MaxTiler

device = torch.device("cuda")
torch.set_default_device(device)

state_dict = torch.load("spanplus_2x.pth")
wrapped_model = global_registry.load_from_state_dict(state_dict)

img = read("test.jpg", None, 0)
tiler = MaxTiler()

output_img = upscale_with_tiler(img, tiler, wrapped_model, device)
save(output_img, "output.png")
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
* [MoSR](https://github.com/umzi2/MoSR)

## Credits
* Based on [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel)

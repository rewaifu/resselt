# resselt

## Usage

```py
import torch
from resselt import load_from_file, load_from_state_dict

# Load model from file
model = load_from_file("spanplus_2x.pth")

# or from state dict
state_dict = torch.load("spanplus_2x.pth")
model = load_from_state_dict(state_dict)

# and after you can get model metadata: ModelMetadata(name, in_channels, out_channels, upscale)
print(model.parameters_info)
```

## Credits
* Based on [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel)

from typing import Mapping

from .arch import GateR
from ...factory import KeyCondition, Architecture
from ...utilities.state_dict import get_seq_len


class GateRArch(Architecture[GateR]):
    def __init__(self):
        super().__init__(
            uid='GateR',
            detect=KeyCondition.has_all(
                'dec0.0.bias',
                'dec0.0.weight',
                'dec0.1.gated.0.conv.conv.bias',
                'dec0.1.gated.0.conv.conv.weight',
                'dec0.1.gated.0.fc1.bias',
                'dec0.1.gated.0.fc1.weight',
                'dec0.1.gated.0.fc2.bias',
                'dec0.1.gated.0.fc2.weight',
                'dec0.1.gated.0.norm.weight',
                'dec0.2.body.0.bias',
                'dec0.2.body.0.weight',
                'dec1.0.bias',
                'dec1.0.weight',
                'dec1.1.gated.0.conv.conv.bias',
                'dec1.1.gated.0.conv.conv.weight',
                'dec1.1.gated.0.fc1.bias',
                'dec1.1.gated.0.fc1.weight',
                'dec1.1.gated.0.fc2.bias',
                'dec1.1.gated.0.fc2.weight',
                'dec1.1.gated.0.norm.weight',
                'dec1.2.body.0.bias',
                'dec1.2.body.0.weight',
                'dec2.0.gated.0.conv.conv.bias',
                'dec2.0.gated.0.conv.conv.weight',
                'dec2.0.gated.0.fc1.bias',
                'dec2.0.gated.0.fc1.weight',
                'dec2.0.gated.0.fc2.bias',
                'dec2.0.gated.0.fc2.weight',
                'dec2.0.gated.0.norm.weight',
                'dim_to_ch.0.bias',
                'dim_to_ch.0.weight',
                'dim_to_ch.1.bias',
                'dim_to_ch.1.weight',
                'enc0.gated.0.conv.conv.bias',
                'enc0.gated.0.conv.conv.weight',
                'enc0.gated.0.fc1.bias',
                'enc0.gated.0.fc1.weight',
                'enc0.gated.0.fc2.bias',
                'enc0.gated.0.fc2.weight',
                'enc0.gated.0.norm.weight',
                'enc1.0.body.0.bias',
                'enc1.0.body.0.weight',
                'enc1.1.gated.0.conv.conv.bias',
                'enc1.1.gated.0.conv.conv.weight',
                'enc1.1.gated.0.fc1.bias',
                'enc1.1.gated.0.fc1.weight',
                'enc1.1.gated.0.fc2.bias',
                'enc1.1.gated.0.fc2.weight',
                'enc1.1.gated.0.norm.weight',
                'enc2.0.body.0.bias',
                'enc2.0.body.0.weight',
                'enc2.1.gated.0.conv.conv.bias',
                'enc2.1.gated.0.conv.conv.weight',
                'enc2.1.gated.0.fc1.bias',
                'enc2.1.gated.0.fc1.weight',
                'enc2.1.gated.0.fc2.bias',
                'enc2.1.gated.0.fc2.weight',
                'enc2.1.gated.0.norm.weight',
                'in_to_dim.bias',
                'in_to_dim.weight',
                'latent.0.body.0.bias',
                'latent.0.body.0.weight',
                'latent.1.gated.0.fc1.bias',
                'latent.1.gated.0.fc1.weight',
                'latent.1.gated.0.fc2.bias',
                'latent.1.gated.0.fc2.weight',
                'latent.1.gated.0.norm.weight',
                'latent.2.body.0.bias',
                'latent.2.body.0.weight',
            ),
        )

    def load(self, state: Mapping[str, object]):
        block_list = ['enc0', 'enc1.1', 'enc2.1', 'latent.1', 'dec0.1', 'dec1.1', 'dec2.0']
        dim, in_ch = state['in_to_dim.weight'].shape[:2]
        num_blocks = [get_seq_len(state, block + '.gated') for block in block_list]
        latent_att = 'latent.1.gated.0.conv.conv.weight' not in state
        model = GateR(in_ch=in_ch, dim=dim, num_blocks=num_blocks, latent_att=latent_att)

        return self._enhance_model(model=model, in_channels=in_ch, out_channels=int(in_ch), upscale=1, name='GateR')

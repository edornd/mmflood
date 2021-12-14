from typing import Type

import timm
from timm.models.features import FeatureInfo
from torch import Tensor, nn

from floods.models import Encoder
from floods.models.modules import MultimodalAdapter

# implement here other custom encoders if required
# just a simple wrapper to include custom encoders into the list
available_encoders = {name: timm.create_model for name in timm.list_models()}


class MultiEncoder(Encoder):
    def __init__(self,
                 encoder_sar: Encoder,
                 encoder_dem: Encoder,
                 act_layer: Type[nn.Module],
                 norm_layer: Type[nn.Module],
                 return_features: bool = False):
        super().__init__()
        self.encoder_sar = encoder_sar
        self.encoder_dem = encoder_dem
        self.return_features = return_features
        self.ssmas = nn.ModuleList()
        for sar_chs, dem_chs in zip(self.encoder_sar.feature_info.channels(), self.encoder_dem.feature_info.channels()):
            self.ssmas.append(
                MultimodalAdapter(sar_channels=sar_chs,
                                  dem_channels=dem_chs,
                                  act_layer=act_layer,
                                  norm_layer=norm_layer))

    @property
    def feature_info(self) -> FeatureInfo:
        # We use the feature information from the SAR encoder, given that we intend to merge
        # We could have used the DEM, it's equivalent in this case
        return self.encoder_sar.feature_info

    def forward(self, inputs: Tensor) -> Tensor:
        # expecting x to be [batch, 4, h, w]
        # we pass the first 3 to the RGB enc., the last one to the IR enc.
        sar, dem = inputs[:, :-1], inputs[:, -1].unsqueeze(1)
        rgb_features = self.encoder_sar(sar)
        ir_features = self.encoder_dem(dem)
        out_features = []
        for module, sar, dem in zip(self.ssmas, rgb_features, ir_features):
            out_features.append(module(sar, dem))
        return out_features

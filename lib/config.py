"""
Настройки сетей
"""

from typing import List
from dataclasses import dataclass

@dataclass
class ConfigNet:
    backbone: str
    return_layers: dict
    out_channels: int
    in_channels_list: List[int]
    use_last_layer: bool

cfg_re50 = {
    "backbone": "resnet50",
    "return_layers": {"layer1": 1, "layer2": 2, "layer3": 3, "layer4": 4},
    'out_channels': 256,
    "in_channels_list": [256, 512, 1024, 2048],
    "use_last_layer": True,
}
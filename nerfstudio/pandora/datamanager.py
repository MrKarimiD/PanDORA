"""
Pandora HDR DataManager
"""

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.pandora.dataset import (
    HDRInputDataset,
    HDRNerfactoInputDataset,
    HDRNerfactoWoCrfInputDataset,
)


@dataclass
class HDRNerfactoDataManagerConfig(VanillaDataManagerConfig):
    """A HDR data manager, based on VanillaDataManager"""

    _target: Type = field(default_factory=lambda: VanillaDataManager[HDRNerfactoInputDataset])
    """Target class to instantiate."""
    
@dataclass
class HDRNerfactoWoCrfDataManagerConfig(VanillaDataManagerConfig):
    """A HDR data manager, based on VanillaDataManager"""

    _target: Type = field(default_factory=lambda: VanillaDataManager[HDRNerfactoWoCrfInputDataset])
    """Target class to instantiate."""
    

@dataclass
class HDRVanillaDataManagerConfig(VanillaDataManagerConfig):
    """A HDR data manager, based on VanillaDataManager"""

    _target: Type = field(default_factory=lambda: VanillaDataManager[HDRInputDataset])
    """Target class to instantiate."""
    
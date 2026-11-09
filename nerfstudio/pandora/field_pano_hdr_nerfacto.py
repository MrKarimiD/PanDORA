"""
HDR Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions
from nerfstudio.fields.nerfacto_field import \
    NerfactoField  # for subclassing NerfactoField


class PanoHDR_NerfactoField(NerfactoField):
    """Template Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
    """

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = True,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        use_appearance_embedding: bool = True,
    ) -> None:
        super().__init__(aabb, num_images, num_layers, hidden_dim,
                         geo_feat_dim, num_levels, base_res, 
                         max_res, log2_hashmap_size, num_layers_color,
                         num_layers_transient, features_per_level, hidden_dim_color,
                         hidden_dim_transient, appearance_embedding_dim, transient_embedding_dim,
                         use_transient_embedding, use_semantics, num_semantic_classes,
                         pass_semantic_gradients, use_pred_normals, use_average_appearance_embedding,
                         spatial_distortion, implementation, use_appearance_embedding)
        
        if self.use_appearance_embedding:
            self.mlp_head = MLP(
                in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim= 3,
                activation=nn.ReLU(),
                out_activation=nn.ReLU(),
                # out_activation=nn.Sigmoid(),
                implementation=implementation,
            )

        # Define the last part MLP for RGB outputs and Masks
        else:
            self.mlp_head = MLP(
                in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim= 3,
                activation=nn.ReLU(),
                out_activation=nn.ReLU(),
                # out_activation=nn.Sigmoid(),
                implementation=implementation,
            )
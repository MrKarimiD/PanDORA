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


class PandoraNerfactoField(NerfactoField):
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
        predicts_validity: bool = True,
        second_step: bool = False,
    ) -> None:
        super().__init__(aabb, num_images, num_layers, hidden_dim,
                         geo_feat_dim, num_levels, base_res, 
                         max_res, log2_hashmap_size, num_layers_color,
                         num_layers_transient, features_per_level, hidden_dim_color,
                         hidden_dim_transient, appearance_embedding_dim, transient_embedding_dim,
                         use_transient_embedding, use_semantics, num_semantic_classes,
                         pass_semantic_gradients, use_pred_normals, use_average_appearance_embedding,
                         spatial_distortion, implementation, use_appearance_embedding, predicts_validity)
        
        if self.use_appearance_embedding:
            self.mlp_head = MLP(
                in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim= 3,
                activation=nn.ReLU(),
                # out_activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )

            self.mlp_head_fast = MLP(
                in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim= 3,
                activation=nn.ReLU(),
                # out_activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )

        # Define the last part MLP for RGB outputs and Masks
        else:
            self.mlp_head = MLP(
                in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim= 6,
                activation=nn.ReLU(),
                # out_activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )

            self.mlp_head_fast = MLP(
                in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
                num_layers=num_layers_color,
                layer_width=hidden_dim_color,
                out_dim= 3,
                activation=nn.ReLU(),
                # out_activation=nn.ReLU(),
                out_activation=nn.Sigmoid(),
                implementation=implementation,
            )
        self.second_step = second_step

    
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        # if self.second_step:
        #     with torch.no_grad():
        #         density, base_mlp_out = super().get_density(ray_samples)
        # else:
        density, base_mlp_out = super().get_density(ray_samples)
        return density, base_mlp_out
    
    
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.use_appearance_embedding:
            if self.training:
                embedded_appearance = self.embedding_appearance(camera_indices)
            else:
                if self.use_average_appearance_embedding:
                    embedded_appearance = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    ) * self.embedding_appearance.mean(dim=0)
                else:
                    embedded_appearance = torch.zeros(
                        (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                    )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        if self.use_appearance_embedding:
            h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_appearance.view(-1, self.appearance_embedding_dim),
                ],
                dim=-1,
            )
        else:
            h = torch.cat(
                [
                    d,
                    density_embedding.view(-1, self.geo_feat_dim),
                ],
                dim=-1,
            )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        rgb_fast = self.mlp_head_fast(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB_FAST: rgb_fast})

        if self.predicts_validity:
            h_no_grad = h.clone()
            validity = self.mlp_validity(h_no_grad).view(*outputs_shape, -1).to(directions)
            outputs.update({FieldHeadNames.VALIDITY: validity})
        
        return outputs

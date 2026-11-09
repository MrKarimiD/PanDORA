"""
Pandora Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.pandora.field import PandoraNerfactoField
from nerfstudio.pandora.renderer import RGBRenderer_HDR
from nerfstudio.model_components.losses import (
    MSELoss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import (
    ProposalNetworkSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    NormalsRenderer,
    SemanticRenderer,
)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.nerfacto import (  # for subclassing Nerfacto model
    NerfactoModel,
    NerfactoModelConfig,
)
from nerfstudio.utils import colormaps

FAST_EXPOSURE_CUTOFF = 0.27 # good for real data (0.1 for synthetic)
WELL_EXPOSURE_CUTOFF = 0.95 # good for real data (0.9 for synthetic)

WELL_EXPOSURE = 1.0
FAST_EXPOSURE = 0.004

@dataclass
class PandoraModelConfig(NerfactoModelConfig):
    """Template Model Configuration.

    Add your custom model config parameters here.
    """

    _target: Type = field(default_factory=lambda: PandoraModel)

    appearance_embed_dim: int = 32

    pandora_steps: int = -1
    """  Define which steps of pandora is going on!! """

    apply_mu_law: bool = True
    """  Define which steps of pandora is going on!! """


class PandoraModel(NerfactoModel):
    """Template Model."""

    config: PandoraModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Fields
        self.field = PandoraNerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
            second_step=(self.config.pandora_steps == 2)
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer_HDR(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_normals = NormalsRenderer()
        self.renderer_validity = SemanticRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # TODO losses
        self.rgb_loss = MSELoss()

        # metrics, psnr
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)


    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)
        validity = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.VALIDITY], weights=weights)
        rgb_fast = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB_FAST], weights=weights)

        # if self.config.apply_mu_law:
        #     u = 5000.
        #     rgb_fast = torch.exp(rgb_fast * torch.log(torch.tensor(u+1.))) - 1.
        #     rgb_fast /= u
       
        outputs = {
            "rgb": rgb,
            "rgb_fast": rgb_fast,
            "accumulation": accumulation,
            "depth": depth,
            "validity_w": validity[:, 1],
            "validity_f": validity[:, 0],
        }
        
        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs
    
    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
        ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        
        if batch["exposure"] != 1.0: # change to == for testing fast-exposed
            predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        else:
            # if self.config.pandora_steps == 1:
            return None, None # comment for testing fast-exposed
            predicted_rgb = outputs["rgb_fast"]  # Blended with background (black if random background)
            # if self.config.apply_mu_law:
            #     from nerfstudio.utils.colormaps import ColormapOptions, Colormaps
            #     cm = ColormapOptions('inverse-mu')
            #     predicted_rgb = colormaps.apply_colormap(predicted_rgb, cm)
            
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )
        gt_rgb = batch["mask"] * gt_rgb
        predicted_rgb = batch["mask"] * predicted_rgb
        print("Exposure: ", batch['exposure'])

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        # print("combined_rgb: ", combined_rgb.shape, "gt_rgb: ", gt_rgb.shape, "predicted_rgb: ", predicted_rgb.shape)
        combined_rgb = predicted_rgb # torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        # gt_rgb_tonemapped = torch.clamp(colormaps.apply_colormap(gt_rgb), 0, 1)
        gt_rgb_tonemapped = torch.clamp(gt_rgb, 0, 1)
        # predicted_rgb_tonemapped = torch.clamp(colormaps.apply_colormap(predicted_rgb), 0, 1)
        predicted_rgb_tonemapped = torch.clamp(predicted_rgb, 0, 1)
        predicted_rgb_tonemapped = torch.nan_to_num(predicted_rgb_tonemapped, nan=0.0)
        gt_rgb_tonemapped = torch.moveaxis(gt_rgb_tonemapped, -1, 0)[None, ...]
        predicted_rgb_tonemapped = torch.moveaxis(predicted_rgb_tonemapped, -1, 0)[None, ...]
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        # tonemapping before computing lpips
        lpips = self.lpips(gt_rgb_tonemapped, predicted_rgb_tonemapped)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}
        # images_dict = {"img": predicted_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict


    def hat_weighting_function(self, pixels, zmin, zmax):
        threshold = 0.5 * (zmin + zmax)
        weights = torch.zeros(pixels.shape, dtype = pixels.dtype).to(self.device)
        weights = weights + torch.finfo(torch.float32).eps
        
        # mask_lowerbound = zmin <= pixels <= threshold
        mask_lowerbound = torch.ge(pixels, zmin) & torch.le(pixels, threshold)
        weights[mask_lowerbound] = 2.0 * (pixels[mask_lowerbound] - zmin)

        # mask_upperbound = threshold <= pixels <= zmax
        mask_upperbound = torch.ge(pixels, threshold) & torch.le(pixels, zmax)
        weights[mask_upperbound] = 2.0 * (zmax - pixels[mask_upperbound])

        return weights
        
    
    def cut_weighting_function(self, pixels, exposures):
        weights = torch.zeros(pixels.shape, dtype = pixels.dtype).to(self.device)
        
        well_expo_valids = (exposures == 1.0).repeat(1,pixels.shape[-1])
        weights[well_expo_valids] = torch.clip(-20.0 * (pixels[well_expo_valids] - WELL_EXPOSURE_CUTOFF) + 1, 0.0, 1.0)

        fast_expo_valids = ~well_expo_valids
        weights[fast_expo_valids] = torch.clip(10.0 * (pixels[fast_expo_valids] - FAST_EXPOSURE_CUTOFF), 0.0, 1.0)

        return weights


    def cut_weighting_for_mask_function(self, exposures):
        weights_f = torch.zeros(exposures.shape, dtype = exposures.dtype).to(self.device)
        weights_w = torch.zeros(exposures.shape, dtype = exposures.dtype).to(self.device)
        
        well_expo_valids = (exposures == 1.0).repeat(1, 1)
        weights_w[well_expo_valids] = 1.0
        
        fast_expo_valids = ~well_expo_valids
        weights_f[fast_expo_valids] = 1.0

        return weights_f, weights_w
    
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        weights_for_RGB_loss = None

        image = batch["image"].to(self.device)

        if "exposure" in batch:
            exposures_resized = batch["exposure"].view(batch["exposure"].shape[0], 1)
            # img_uncompress_re_exposed =  exposures_resized * img_uncompress
            
        #     # # Debevec Hat weights for Loss
        #     # # weights_for_loss = self.hat_weighting_function(torch.clip(img_uncompress_re_exposed, 0, 1), 0.0, 1.0)
            
        #     # # Clear cut weights for Loss
        #     # weights_for_RGB_loss = self.cut_weighting_function(torch.clip(img_uncompress_re_exposed, 0, 1), exposures_resized)
            mask_f_w, mask_w_w = self.cut_weighting_for_mask_function(exposures_resized)

        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        pred_rgb_f, gt_rgb_f = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb_fast"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        # MSE loss for mask with weights
        # if "saturation_mask" in batch:
        #     mask_in_float = batch['saturation_mask'].type(torch.float).to(self.device)
        #     negative_mask_in_float = torch.ones(mask_in_float.shape).to(self.device) - mask_in_float
        #     loss_mask_fast_expo = (mask_f_w * (( 10.0 * mask_in_float - torch.unsqueeze(outputs["validity_f"], 1)) ** 2)).mean()
        #     # loss_mask_fast_expo = (mask_f_w * (( mask_in_float - torch.unsqueeze(outputs["validity_f"], 1)) ** 2)).mean()
        #     loss_mask_well_expo = (mask_w_w * (( mask_in_float - torch.unsqueeze(outputs["validity_w"], 1)) ** 2)).mean()
        #     loss_dict["validity_loss_well_exposed"] = loss_mask_well_expo # 0.01
        #     loss_dict["validity_loss_fast_exposure"] = loss_mask_fast_expo

        assert self.config.pandora_steps == 1 or self.config.pandora_steps == 2, "The step of Pandora should be either 1 or 2!! "
        
        if self.config.pandora_steps == 1:
            loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        else:
            gt_rgb_w = gt_rgb.clone()
            
            gt_rgb_f = gt_rgb.clone()
            if self.config.apply_mu_law:
                u = 5000.
                gt_rgb_f = torch.log(1. + u * gt_rgb_f) / torch.log(torch.tensor(1.+u))

            loss_dict["rgb_loss"] = (mask_w_w * ((gt_rgb_w - pred_rgb) ** 2)).mean()
            loss_dict["rgb_loss_fast"] = (mask_f_w * ((gt_rgb_f - pred_rgb_f) ** 2)).mean()

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            # loss_dict["fg"] = (torch.pow(1 - torch.sum( outputs["weights_list"][-1], dim=0), 2)).mean()
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict
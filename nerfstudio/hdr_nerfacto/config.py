
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.hdr_nerfacto.hdr_nerf_model import HdrNerfactoModelConfig
from nerfstudio.pandora.datamanager import HDRNerfactoDataManagerConfig, HDRNerfactoWoCrfDataManagerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig


def get_hdr_nerfacto_config(*, use_crf=True, clip_before_accumulation=False):
    if use_crf:
        method_name = "hdr-nerfacto"
        assert not clip_before_accumulation
    else:
        if clip_before_accumulation:
            method_name = "hdr-nerfacto-wo-crf-clip-bf-acc"
        else:
            method_name = "hdr-nerfacto-wo-crf"
    
    datamanager_config_class = HDRNerfactoDataManagerConfig if use_crf else HDRNerfactoWoCrfDataManagerConfig
        
    return TrainerConfig(
        method_name=method_name,
        steps_per_eval_batch=500,
        steps_per_save=2000,
        # max_num_iterations=1000,
        max_num_iterations=60000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=datamanager_config_class(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                camera_optimizer=CameraOptimizerConfig(
                    mode="off"
                ),
            ),
            model=HdrNerfactoModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_crf=use_crf,
                clip_before_accumulation=clip_before_accumulation,
            ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    )

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import CosineDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from livescene.datamanager.livescene_datamanager import LiveSceneDataManagerConfig
from livescene.datamanager.livescene_dataparser import LiveSceneSyntheticDataParserConfig
from livescene.livescene_controller import LiveSceneControllerConfig
from livescene.livescene_model import LiveSceneModelConfig
from livescene.livescene_pipeline import LiveScenePipelineConfig

livescene_method = MethodSpecification(
    config=TrainerConfig(
        method_name="livescene",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_image=500,
        steps_per_eval_all_images=30000,
        max_num_iterations=30001,
        mixed_precision=True,
        pipeline=LiveScenePipelineConfig(
            datamanager=LiveSceneDataManagerConfig(
                dataparser=LiveSceneSyntheticDataParserConfig(
                    scale_factor=1,
                    scene_scale=2,
                ),
                train_num_rays_per_batch=1 << 12,
                eval_num_rays_per_batch=1 << 12,
            ),
            model=LiveSceneModelConfig(
                use_same_proposal_network=False,
                num_proposal_iterations=2,
                near_plane=0.01,
                far_plane=6.0,
                background_color="white",
                eval_num_rays_per_chunk=1 << 16,
                num_samples=48,
                grid_base_resolution=[128, 128, 128, 32],
                grid_feature_dim=32,
                multiscale_res=[1, 2, 4],
                proposal_net_args_list=[
                    {"density_feature_dim": 8, "density_resolution": [64, 64, 64, 32]},
                    {"density_feature_dim": 8, "density_resolution": [128, 128, 128, 32]},
                ],
                proposal_lang_net_args_list=[
                    {
                        "lang_feature_dim": 8,
                        "lang_resolution": [128, 128, 128],
                        "max_prob": 0.0,
                        "use_pe": True,
                        "val_resolution": 256,
                        "val_feature_dim": 4,
                        "val_neus": 32,
                        "cat_embed": False,
                        "clip_val": False,
                        "lang_nlys": 1,
                        "lang_neus": 64,
                    },
                ],
                loss_coefficients={
                    "rgb": 1.0,
                    "interlevel": 1.0,
                    "distortion": 0.001,
                    "plane_tv": 0.1,
                    "plane_tv_proposal_net": 0.0001,
                    "l1_time_planes": 0.001,
                    "l1_time_planes_proposal_net": 0.0001,
                    "time_smoothness": 0.1,
                    "time_smoothness_proposal_net": 0.001,
                    "plane_tv_proposal_language_net": 0.0001,
                    "ray_mask_loss": 0.001,
                    "ray_clip_loss": 0.1,
                    "plane_tv_val_embeds": 0.0001,
                    "embed_val_mse": 0.001,
                    "repulsion_loss": 0.001,
                },
                lang_ws=[1.0, 1.0, 1.0],
            ),
            controller=LiveSceneControllerConfig(
                clip_type="openclip",
                clip_args={"clip_model_type": "ViT-B/16", "clip_n_dims": 512},
                openclip_args={
                    "clip_model_type": "ViT-B-16",
                    "clip_model_pretrained": "laion2b_s34b_b88k",
                    "clip_n_dims": 512,
                },
            ),
            use_gt_val=True,
            train_with_lang=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "proposal_language_networks": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=0.02, eps=1e-12),
                "scheduler": CosineDecaySchedulerConfig(warm_up_end=512, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 16),
        vis="viewer",
    ),
    description="Livescene model for dynamic scenes with lang control",
)

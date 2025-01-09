"""
LiveScene Model
Adapted from kplanes_nerfstudio.kplanes.kplanes
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import cv2
import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import (
    # ProposalNetworkSampler,
    UniformLinDispPiecewiseSampler,
    UniformSampler,
)
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer, SemanticRenderer
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import misc
from nerfstudio.utils.colormaps import apply_colormap, apply_depth_colormap
from scipy.ndimage import gaussian_filter
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from livescene.livescene_controller import LiveSceneController
from livescene.livescene_field import LiveSceneDensityField, LiveSceneField, LiveSceneLanguageField
from livescene.ray_utils import ProposalNetworkSampler


@dataclass
class LiveSceneModelConfig(ModelConfig):
    """LiveScene Model Config"""

    _target: Type = field(default_factory=lambda: LiveSceneModel)

    near_plane: float = 0.05
    """How far along the ray to start sampling."""

    far_plane: float = 2.0
    """How far along the ray to stop sampling."""

    grid_base_resolution: List[int] = field(default_factory=lambda: [128, 128, 128])
    """Base grid resolution."""

    grid_feature_dim: int = 32
    """Dimension of feature vectors stored in grid."""

    multiscale_res: List[int] = field(default_factory=lambda: [1, 2, 4])
    """Multiscale grid resolutions."""

    is_contracted: bool = False
    """Whether to use scene contraction (set to true for unbounded scenes)."""

    concat_features_across_scales: bool = True
    """Whether to concatenate features at different scales."""

    linear_decoder: bool = False
    """Whether to use a linear decoder instead of an MLP."""

    linear_decoder_layers: Optional[int] = 1
    """Number of layers in linear decoder"""

    # * proposal sampling arguments
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""

    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""

    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"density_feature_dim": 8, "density_resolution": [64, 64, 64]},
            {"density_feature_dim": 8, "density_resolution": [128, 128, 128]},
        ]
    )

    proposal_lang_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"lang_feature_dim": 8, "lang_resolution": [64, 64, 64]},
            {"lang_feature_dim": 8, "lang_resolution": [128, 128, 128]},
        ]
    )

    """* Arguments for the proposal density fields."""

    num_proposal_samples: Optional[Tuple[int, ...]] = (256, 128)
    """ * Number of samples per ray for each proposal network.* """

    num_samples: Optional[int] = 48
    """Number of samples per ray used for rendering."""

    single_jitter: bool = False
    """Whether use single jitter or not for the proposal networks."""

    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps."""

    proposal_update_every: int = 5
    """Sample every n steps after the warmup."""

    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""

    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""

    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""

    appearance_embedding_dim: int = 0
    """Dimension of appearance embedding. Set to 0 to disable."""

    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""

    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """The background color as RGB."""

    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
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
        }
    )
    """Loss coefficients."""

    lang_ws: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    """the weights of the language network"""

    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""

    collider: str = "aabb"
    """Collider type to use: aabb | nearfar"""

    repuls_delta: float = 0.5
    """repuls loss params delta """

    repuls_ratio: float = 0.5
    """repuls loss params sampling ratio"""

    repuls_thresh: float = 10.0
    """repuls loss params thresh"""


class LiveSceneModel(Model):
    config: LiveSceneModelConfig
    """LiveScene model

    Args:
        config: LiveScene configuration to instantiate model
    """

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.is_contracted:
            scene_contraction = SceneContraction(order=float("inf"))
        else:
            scene_contraction = None

        # controller
        self.controller: LiveSceneController = self.kwargs["viewer_controller"]

        # * == Fields ==
        self.field = LiveSceneField(
            self.scene_box.aabb,
            num_images=self.num_train_data,
            grid_base_resolution=self.config.grid_base_resolution,
            grid_feature_dim=self.config.grid_feature_dim,
            concat_across_scales=self.config.concat_features_across_scales,
            multiscale_res=self.config.multiscale_res,
            spatial_distortion=scene_contraction,
            appearance_embedding_dim=self.config.appearance_embedding_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            linear_decoder=self.config.linear_decoder,
            linear_decoder_layers=self.config.linear_decoder_layers,
        )

        self.density_fns, self.lang_fns = [], []
        num_prop_nets = self.config.num_proposal_iterations

        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        self.proposal_language_networks = torch.nn.ModuleList()

        prop_lang_net_args = self.config.proposal_lang_net_args_list[-1]
        prop_lang_net_args["clip_n_dims"] = self.controller.clip_encoder.clip_n_dims
        prop_lang_net_args["max_fids"] = self.kwargs["metadata"]["ids"].shape[0]
        prop_lang_net_args["use_gt_val"] = self.kwargs["use_gt_val"]
        prop_lang_net_args["train_with_lang"] = self.kwargs["train_with_lang"]

        if self.config.use_same_proposal_network:
            prop_net_args = self.config.proposal_net_args_list[-1]

            network = LiveSceneDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, linear_decoder=self.config.linear_decoder, **prop_net_args)
            lang_networks = LiveSceneLanguageField(self.scene_box.aabb, spatial_distortion=scene_contraction, linear_decoder=self.config.linear_decoder, **prop_lang_net_args)

            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])

            self.proposal_language_networks.append(lang_networks)
            self.lang_fns.extend([lang_networks.lang_fn for _ in range(num_prop_nets + 1)])  # * +1 for the final PDF
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = LiveSceneDensityField(self.scene_box.aabb, spatial_distortion=scene_contraction, linear_decoder=self.config.linear_decoder, **prop_net_args)
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

            lang_networks = LiveSceneLanguageField(self.scene_box.aabb, spatial_distortion=scene_contraction, linear_decoder=self.config.linear_decoder, **prop_lang_net_args)
            for i in range(num_prop_nets + 1):  # * +1 for the final PDF
                self.proposal_language_networks.append(lang_networks)
            self.lang_fns.extend([lang_networks.lang_fn for lang_networks in self.proposal_language_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]), 1, self.config.proposal_update_every)

        if self.config.is_contracted:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=self.config.single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=self.config.single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_samples,
            num_proposal_samples_per_ray=self.config.num_proposal_samples,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        if self.config.collider == "aabb":
            self.collider = AABBBoxCollider(scene_box=self.scene_box, near_plane=self.config.near_plane)
        elif self.config.collider == "nearfar":
            self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)
        else:
            raise ValueError(f"Invalid collider type: {self.config.collider}")

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
        self.renderer_mask = SemanticRenderer()
        self.renderer_clip = SemanticRenderer()

        # losses
        self.rgb_loss = MSELoss()
        self.embed_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.temporal_distortion = len(self.config.grid_base_resolution) == 4  # for viewer

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {
            "proposal_networks": list(self.proposal_networks.parameters()),
            "proposal_language_networks": list(self.proposal_language_networks.parameters()),
            "fields": list(self.field.parameters()),
        }
        return param_groups

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        lang_fns = self.lang_fns
        if "cam_idx" in ray_bundle.metadata:
            if self.kwargs["train_with_lang"]:
                id = self.kwargs["metadata"]["ids"].to(self.device).squeeze()
                clip_net = self.proposal_language_networks[0].semantic_mapping
                embed = clip_net(id).reshape(-1, clip_net.get_out_dim())
                atrb_val = self.controller.get_atrb_vals(embed)
                ray_bundle.metadata["atrb_val"] = atrb_val
            else:
                atrb_val = self.controller.get_slider_vals()
                ray_bundle.metadata["atrb_val"] = torch.tensor(atrb_val, device=self.device)[None, :]
            lang_fns = [functools.partial(f, viewer=True) for f in self.lang_fns]
        ray_samples, weights_list, ray_samples_list, probability_list, clip_feat_list, query_value_list = self.proposal_sampler(ray_bundle, lang_fns=lang_fns, density_fns=self.density_fns)

        field_outputs = self.field(ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth}

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list  # [num_proposal_iterations+1]
            outputs["ray_samples_list"] = ray_samples_list  # [num_proposal_iterations+1]
            outputs["probability_list"] = probability_list  # [num_proposal_iterations+1]
            # outputs["clip_feat_list"] = clip_feat_list  # [num_proposal_iterations+1]
            outputs["query_value_list"] = query_value_list  # [num_proposal_iterations+1]

        for i in range(self.config.num_proposal_iterations + 1):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
            outputs[f"prop_mask_{i}"] = self.renderer_mask(probability_list[i], weights=weights_list[i].detach())  # (num_rays, num_samples, C)

            if self.training:
                if clip_feat_list[i] is None:
                    continue
                if i == self.config.num_proposal_iterations:
                    outputs[f"prop_clip_{i}"] = self.renderer_clip(clip_feat_list[i], weights=weights_list[i].detach())  # (num_rays, num_samples, clip_dim)
        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)
        for key, value in outputs.copy().items():
            if "mask" in key:
                outputs.pop(key)
                if str(self.config.num_proposal_iterations) in key:
                    value = torch.softmax(value, -1)
                    for i in range(value.shape[-1] - 1):
                        outputs[f"atrb_{i}_raw"] = value[..., i : i + 1]
                        outputs[f"composited_{i}"] = custom_colormap(outputs["rgb"], outputs[f"atrb_{i}_raw"].squeeze(), alpha=0.5)

        outputs["atrb_raw"] = torch.stack([outputs[f"atrb_{i}_raw"] for i in range(value.shape[-1] - 1)], dim=0)
        outputs["atrb_raw"] = torch.max(outputs["atrb_raw"], dim=0)[0]
        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)  # Blend if RGBA

        metrics_dict = {"psnr": self.psnr(outputs["rgb"], image)}
        if self.training:
            metrics_dict["interlevel"] = interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

            # * cross entropy loss for rays
            valid = batch["mask_valid"][batch["atrb_mask"].bool()][:, None]
            prop_probability_list = [outputs[f"prop_mask_{i}"] for i in range(self.config.num_proposal_iterations + 1)]
            prop_gt_probability_list = [batch["atrb_mask"] for _ in range(self.config.num_proposal_iterations + 1)]
            prop_valid_list = [valid for _ in range(self.config.num_proposal_iterations + 1)]
            metrics_dict["ray_mask_loss"] = cross_entropy_loss(prop_probability_list, prop_gt_probability_list, valid_list=prop_valid_list, ws=self.config.lang_ws)

            # * repulsion_loss
            metrics_dict["repulsion_loss"] = repulsion_loss(
                prop_probability_list, prop_gt_probability_list, valid_list=prop_valid_list, delta=self.config.repuls_delta, s=self.config.repuls_ratio, thresh=self.config.repuls_thresh
            )

            # * last layer clip loss
            if f"prop_clip_{self.config.num_proposal_iterations}" in outputs:
                prop_clip_feat_list = [outputs[f"prop_clip_{self.config.num_proposal_iterations}"]]
                prop_gt_clip_feat_list = [batch["clip"]]
                prop_valid_list = [valid]
                metrics_dict["ray_clip_loss"] = huber_loss(prop_clip_feat_list, prop_gt_clip_feat_list, valid_list=prop_valid_list)

            # * last layer value mse loss
            if not self.kwargs["use_gt_val"]:
                gt_atrb_val = batch["atrb_val"][..., :-1]
                m = batch["atrb_val_mask"][..., :-1]
                metrics_dict["embed_val_mse"] = self.embed_loss(gt_atrb_val[m], outputs["query_value_list"][-1][:, 0, :][m])

            prop_grids = [p.grids.plane_coefs for p in self.proposal_networks]
            prop_lang_grids = [p.grids.plane_coefs for p in self.proposal_language_networks]
            field_grids = [g.plane_coefs for g in self.field.grids]

            metrics_dict["plane_tv"] = space_tv_loss(field_grids)
            metrics_dict["plane_tv_proposal_net"] = space_tv_loss(prop_grids)
            metrics_dict["plane_tv_proposal_language_net"] = space_tv_loss(prop_lang_grids)

            if not self.kwargs["use_gt_val"]:
                prop_val_embeds = [p.val_embeds.plane_coefs for p in self.proposal_language_networks]
                metrics_dict["plane_tv_val_embeds"] = space_tv_loss(prop_val_embeds, force_only_w=True)

            if len(self.config.grid_base_resolution) == 4:
                metrics_dict["l1_time_planes"] = l1_time_planes(field_grids)
                metrics_dict["l1_time_planes_proposal_net"] = l1_time_planes(prop_grids)
                # metrics_dict["l1_time_planes_proposal_language_net"] = l1_time_planes(prop_lang_grids)

                metrics_dict["time_smoothness"] = time_smoothness(field_grids)
                metrics_dict["time_smoothness_proposal_net"] = time_smoothness(prop_grids)
                # metrics_dict["time_smoothness_proposal_language_net"] = time_smoothness(prop_lang_grids)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"].to(self.device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )  # (num_rays, C)

        loss_dict = {"rgb": self.rgb_loss(image, pred_rgb) * self.config.loss_coefficients["rgb"]}

        if self.training:
            for key in self.config.loss_coefficients:
                if key in metrics_dict:
                    loss_dict[key] = metrics_dict[key].clone()
            loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)

        # rgb = outputs["rgb"]
        rgb = torch.clip(outputs["rgb"], min=0.0, max=1.0)
        acc = apply_colormap(outputs["accumulation"])
        depth = apply_depth_colormap(outputs["depth"], accumulation=outputs["accumulation"])

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(self.psnr(image, rgb).item()), "ssim": float(self.ssim(image, rgb)), "lpips": float(self.lpips(image, rgb))}
        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations + 1):
            key = f"prop_depth_{i}"
            prop_depth_i = apply_depth_colormap(outputs[key], accumulation=outputs["accumulation"])
            images_dict[key] = prop_depth_i
        return metrics_dict, images_dict


def compute_plane_tv(t: torch.Tensor, only_w: bool = False) -> float:
    """Computes total variance across a plane.

    Args:
        t: Plane tensor
        only_w: Whether to only compute total variance across w dimension

    Returns:
        Total variance
    """
    _, h, w = t.shape
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).mean()

    if only_w:
        return w_tv

    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).mean()
    return h_tv + w_tv


def space_tv_loss(multi_res_grids: List[torch.Tensor], force_only_w=False, feat_first=True) -> float:
    """Computes total variance across each spatial plane in the grids.

    Args:
        multi_res_grids: Grids to compute total variance over

    Returns:
        Total variance
    """

    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        if len(grids) == 3:
            spatial_planes = {0, 1, 2}
        else:
            spatial_planes = {0, 1, 3}

        for grid_id, grid in enumerate(grids):
            if not feat_first:
                grid = grid.permute(2, 0, 1)
            if grid_id in spatial_planes and not force_only_w:
                total += compute_plane_tv(grid)
            else:
                # Space is the last dimension for space-time planes.
                total += compute_plane_tv(grid, only_w=True)
            num_planes += 1
    return total / num_planes


def cross_entropy_loss(probability_list: List[torch.Tensor], gt_probability_list: List[torch.Tensor], valid_list: List[torch.Tensor], alpha=0.5, gamma=1.5, ws=[1.0, 1.0, 1.0]) -> torch.Tensor:
    """
    probability_list: List[torch.Tensor], shape [num_proposal_iterations, num_rays, num_samples, C], the probability of each sample
    gt_probability_list: List[torch.Tensor], shape [num_proposal_iterations, num_rays, num_samples, C], the ground truth mask
    valid_list: List[torch.Tensor], shape [num_proposal_iterations, num_rays, num_samples, C], the valid mask for gt_probability_list
    """
    if len(probability_list[0].shape) == 3:
        sh_fn = lambda x: x.reshape(-1, x.shape[-1])
    else:
        sh_fn = lambda x: x

    samples_ce_loss = 0.0
    # EPS = torch.finfo(torch.float32).eps
    for i in range(len(probability_list)):
        # * focal loss for multi class classification:
        # * https://discuss.pytorch.org/t/focal-loss-for-imbalanced-multi-class-classification-in-pytorch/61289
        # m = valid_list[i].all(-1)
        m = valid_list[i].squeeze(-1)
        # print(f"** cross_entropy_loss mask {m.all()} **")
        probs_i, gt_probs_i = sh_fn(probability_list[i][m]), sh_fn(gt_probability_list[i][m])
        ce_loss = torch.nn.functional.cross_entropy(probs_i, gt_probs_i, reduction="none")  # (num_rays*num_samples)
        focal_loss = alpha * (1 - torch.exp(-ce_loss)) ** gamma * ce_loss
        samples_ce_loss += focal_loss.mean() * ws[i]
        # samples_ce_loss += torch.sum(focal_loss * sh_fn(valid_list[i])[..., 0]) / (valid_list[i][..., 0].sum() + EPS)  # all C valid mask stay the same
    return samples_ce_loss


def huber_loss(feat_list: List[torch.Tensor], gt_feat_list: List[torch.Tensor], valid_list: List[torch.Tensor], delta=1.25) -> torch.Tensor:
    """
    feat_list: List[torch.Tensor], shape [num_proposal_iterations, num_rays, num_samples, dim]
    gt_feat_list: List[torch.Tensor], shape [num_proposal_iterations, num_rays, num_samples, dim]
    valid_list: List[torch.Tensor], shape [num_proposal_iterations, num_rays, num_samples, 1]
    """
    if len(feat_list[0].shape) == 3:
        sh_fn = lambda x: x.reshape(-1, x.shape[-1])
    else:
        sh_fn = lambda x: x

    samples_huber_loss = 0.0
    for i in range(len(feat_list)):
        m = valid_list[i].squeeze(-1)
        feat_i, gt_feat_i = sh_fn(feat_list[i][m]), sh_fn(gt_feat_list[i][m])
        huber_loss = torch.nn.functional.huber_loss(feat_i, gt_feat_i, delta=delta, reduction="none")
        samples_huber_loss += huber_loss.nansum(dim=-1).nanmean()
    return samples_huber_loss


def l1_time_planes(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes the L1 distance from the multiplicative identity (1) for spatiotemporal planes.

    Args:
        multi_res_grids: Grids to compute L1 distance over

    Returns:
         L1 distance from the multiplicative identity (1)
    """
    time_planes = [2, 4, 5]  # These are the spatiotemporal planes
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        for grid_id in time_planes:
            total += torch.abs(1 - grids[grid_id]).mean()
            num_planes += 1

    return total / num_planes


def compute_plane_smoothness(t: torch.Tensor) -> float:
    """Computes smoothness across the temporal axis of a plane

    Args:
        t: Plane tensor

    Returns:
        Time smoothness
    """
    _, h, _ = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., : h - 1, :]  # [c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., : h - 2, :]  # [c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


def time_smoothness(multi_res_grids: List[torch.Tensor]) -> float:
    """Computes smoothness across each time plane in the grids.
    Args:
        multi_res_grids: Grids to compute time smoothness over

    Returns:
        Time smoothness
    """
    total = 0.0
    num_planes = 0
    for grids in multi_res_grids:
        time_planes = [2, 4, 5]  # These are the spatiotemporal planes
        for grid_id in time_planes:
            total += compute_plane_smoothness(grids[grid_id])
            num_planes += 1

    return total / num_planes


def repulsion_loss(probability_list: List[torch.Tensor], gt_probability_list: List[torch.Tensor], valid_list: List[torch.Tensor], delta=0.5, s=0.5, thresh=10.0) -> torch.Tensor:
    """
    probability_list: List[torch.Tensor], shape [num_proposal_iterations, num_rays, num_samples, C], the probability of each sample
    gt_probability_list: List[torch.Tensor], shape [num_proposal_iterations, num_rays, num_samples, C], the ground truth mask
    valid_list: List[torch.Tensor], shape [num_proposal_iterations, num_rays, num_samples, C], the valid mask for gt_probability_list
    """
    samples_repulsion_loss = 0.0
    for i in range(len(probability_list)):
        m = valid_list[i].squeeze(-1)
        # probs_i, gt_probs_i = torch.softmax(probability_list[i][m], -1), gt_probability_list[i][m]
        probs_i, gt_probs_i = probability_list[i][m], gt_probability_list[i][m]
        n, _ = probs_i.shape
        num = min(int(n * s), n // 2)
        permuted_indices = torch.randperm(n)
        ids1, ids2 = permuted_indices[:num], permuted_indices[num : num * 2]
        mx = gt_probs_i[ids1] != gt_probs_i[ids2]
        fea_dist = torch.nn.functional.l1_loss(probs_i[ids1], probs_i[ids2], reduction="none")[mx].mean()
        # print(f"shape: probs_i: {probs_i.shape}, gt_probs_i: {gt_probs_i.shape}, permuted_indices: {permuted_indices.shape}, ids1: {ids1.shape}, mx: {mx.shape}")
        # samples_repulsion_loss += torch.sigmoid(thresh - fea_dist)
        samples_repulsion_loss += torch.nn.functional.elu(thresh - fea_dist) + 1
        # print(f"feature distance: {fea_dist}, samples_repulsion_loss: {samples_repulsion_loss}")
    return samples_repulsion_loss


def custom_colormap(image, mask, alpha=0.5, thres=0.5):
    mask = mask.detach().cpu().numpy()
    mask = gaussian_filter(mask, sigma=0.4)
    weighted_mask = (np.exp(4 * mask) + np.random.normal(scale=0.1, size=mask.shape))

    device = image.device
    image = image.detach().cpu().numpy()
    weighted_mask[-1, -1] = weighted_mask.max() + weighted_mask.max() * 0.7
    normalized_weighted_mask = cv2.normalize(weighted_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    composite = cv2.applyColorMap(normalized_weighted_mask, cv2.COLORMAP_TURBO)
    composite = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    composite = cv2.addWeighted(composite, alpha, (image * 255.0).astype(np.uint8), 1 - alpha, 0) / 255.0
    composite[mask <= thres] = image[mask <= thres]
    return torch.from_numpy(composite).to(device)
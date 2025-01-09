"""
LiveScene DataManager
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.dataparser_configs import AnnotatedDataParserUnion
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.viewer.server.viewer_elements import Cameras

from livescene.datamanager.clip_utils import embed_clip, embed_clip_avg_views
from livescene.datamanager.livescene_dataparser import LiveSceneRealDataParserConfig
from livescene.livescene_controller import LiveSceneClipEncoder


@dataclass
class LiveSceneDataManagerConfig(VanillaDataManagerConfig):
    """LiveScene DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: LiveSceneDataManager)
    """Target class to instantiate."""
    dataparser: AnnotatedDataParserUnion = LiveSceneRealDataParserConfig()
    """Specifies the dataparser used to unpack the data."""

class LiveSceneDataManager(VanillaDataManager):
    """LiveScene DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: LiveSceneDataManagerConfig

    def __init__(
        self,
        config: LiveSceneDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs)
        self.img_encoder: LiveSceneClipEncoder = kwargs["image_encoder"]
        self.use_gt_val = kwargs["use_gt_val"]
        self.train_with_lang = kwargs["train_with_lang"]
        train_imgs = torch.stack([self.train_dataset[i]["image"].permute(2, 0, 1) for i in range(len(self.train_dataset))])  # (num_img, C, H, W) rgb
        eval_imgs = torch.stack([self.eval_dataset[i]["image"].permute(2, 0, 1) for i in range(len(self.eval_dataset))])  # (num_img, C, H, W) rgb
        if self.train_with_lang:
            train_masks, train_mask_valids = self.train_dataset.metadata["atrb_masks"], self.train_dataset.metadata["mask_valids"]
            val_masks, val_mask_valids = self.eval_dataset.metadata["atrb_masks"], self.eval_dataset.metadata["mask_valids"]
            train_data_index, train_clip_embeddings = embed_clip(self.img_encoder, train_imgs, train_masks.cpu().numpy().astype(np.bool_), train_mask_valids.bool())
            val_data_index, val_clip_embeddings = embed_clip(self.img_encoder, eval_imgs, val_masks.cpu().numpy().astype(np.bool_), val_mask_valids.bool())
            self.train_dataset.metadata.update(
                {
                    "clip_index": train_data_index,
                    "clip_feat": train_clip_embeddings,
                }
            )
            self.eval_dataset.metadata.update(
                {
                    "clip_index": val_data_index,
                    "clip_feat": val_clip_embeddings,
                }
            )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_train(step)
        img_idx, height_idx, width_idx = batch["indices"].cpu().unbind(-1)
        metadata = self.train_dataset.metadata

        batch["atrb_mask"] = metadata["atrb_masks"][img_idx, height_idx, width_idx].to(self.device)  # (num_rays, M)
        batch["mask_valid"] = metadata["mask_valids"][img_idx].to(self.device)  # (num_rays, M)
        batch["atrb_val"] = metadata["atrb_vals"][img_idx].to(self.device)  # (num_rays, M)
        batch["atrb_val_mask"] = metadata["atrb_val_masks"][img_idx].to(self.device)  # (num_rays, M)
        if self.train_with_lang:
            batch["clip"] = metadata["clip_feat"][img_idx][metadata["clip_index"][img_idx, height_idx, width_idx]].to(self.device)  # (num_rays, clip_dim)

        ray_bundle.metadata["id"] = metadata["ids"][img_idx].to(self.device)  # * (num_rays, 1)
        if self.use_gt_val:
            ray_bundle.metadata["atrb_val"] = metadata["atrb_vals"][img_idx].to(self.device)  # * (num_rays, M)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        ray_bundle, batch = super().next_eval(step)
        img_idx, height_idx, width_idx = batch["indices"].cpu().unbind(-1)
        metadata = self.eval_dataset.metadata

        batch["atrb_mask"] = metadata["atrb_masks"][img_idx, height_idx, width_idx].to(self.device)  # (num_rays, M)
        batch["mask_valid"] = metadata["mask_valids"][img_idx].to(self.device)  # (num_rays, M)
        batch["atrb_val"] = metadata["atrb_vals"][img_idx].to(self.device)  # (num_rays, M)
        batch["atrb_val_mask"] = metadata["atrb_val_masks"][img_idx].to(self.device)  # (num_rays, M)
        if self.train_with_lang:
            batch["clip"] = metadata["clip_feat"][img_idx][metadata["clip_index"][img_idx, height_idx, width_idx]].to(self.device)  # (num_rays, clip_dim)

        ray_bundle.metadata["id"] = metadata["ids"][img_idx].to(self.device)  # * (num_rays, 1)
        if self.use_gt_val:
            ray_bundle.metadata["atrb_val"] = metadata["atrb_vals"][img_idx].to(self.device)  # * (num_rays, M)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        camera, batch = super().next_eval_image(step)
        img_idx = int(batch["image_idx"])
        camera.metadata = {
            "id": self.eval_dataset.metadata["ids"][img_idx: img_idx+1].to(self.device),
        }
        if self.use_gt_val:
            camera.metadata["atrb_val"] = self.eval_dataset.metadata["atrb_vals"][img_idx: img_idx+1].to(self.device)
        return camera, batch
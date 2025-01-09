# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Type

import imageio
import numpy as np
import torch
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.blender_dataparser import Blender, BlenderDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

from livescene.datamanager.atrb_utils import read_attributes

MAX_AUTO_RESOLUTION = 1600


@dataclass
class LiveSceneRealDataParserConfig(NerfstudioDataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: LiveSceneReal)
    """target class to instantiate"""
    norm_vals: bool = True
    """whether normalize the attribute values to [0, 1]"""
    interpolate_val: bool = False
    """whether interpolate the attribute values"""
    offset: List[float] = field(default_factory=lambda: [0, 0, 0])



@dataclass
class LiveSceneReal(Nerfstudio):
    """LiveScene DatasetParser for PolyCam"""

    config: LiveSceneRealDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        dataparser_outputs = super()._generate_dataparser_outputs(split)

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        aabb = dataparser_outputs.scene_box.aabb
        dataparser_outputs.scene_box.aabb += aabb[1, :] * torch.Tensor(self.config.offset)[None, :]

        self.max_fid = max([int(Path(frame["file_path"]).stem.split("_")[-1]) for frame in meta["frames"]])
        fids = [Path(name).stem.split("_")[-1] for name in dataparser_outputs.image_filenames]
        ids = torch.tensor([int(fid) / self.max_fid for fid in fids], dtype=torch.float32)[:, None]
        self.num_atrbs, atrb_vals, atrb_val_masks, atrb_masks, mask_valids = read_attributes(fids, data_dir, self.config.norm_vals, not self.config.interpolate_val)

        dataparser_outputs.metadata.update(
            {
                "atrb_vals": atrb_vals,
                "atrb_val_masks": atrb_val_masks,
                "atrb_masks": atrb_masks, 
                "mask_valids": mask_valids,
                "ids": ids,
            }
        )
        return dataparser_outputs


@dataclass
class LiveSceneSyntheticDataParserConfig(BlenderDataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: LiveSceneSynthetic)
    """target class to instantiate"""
    scene_scale: float = 1.5
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    """
    The method to use for splitting the dataset into train and eval. 
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 2
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    norm_vals: bool = True
    """whether normalize the attribute values to [0, 1]"""
    offset: List[float] = field(default_factory=lambda: [0, 0, 0])
    """offset of the bbox"""


@dataclass
class LiveSceneSynthetic(Blender):
    """LiveScene DatasetParser for Omnigibson Behavior"""

    config: LiveSceneSyntheticDataParserConfig

    def __init__(self, config: LiveSceneSyntheticDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        if self.config.data.suffix == ".json":
            meta = load_from_json(self.config.data)
            data_dir = self.config.data.parent
        else:
            meta = load_from_json(self.config.data / "transforms.json")
            data_dir = self.config.data

        image_filenames = []
        depth_filenames = []
        poses = []
        for frame in meta["frames"]:
            image_name = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            depth_name = self.data / Path(frame["file_path"].replace("./images", "depth") + ".npy")
            image_filenames.append(image_name)
            depth_filenames.append(depth_name)
            poses.append(np.array(frame["transform_matrix"]))
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        poses[:, :3, 3] *= self.scale_factor

        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log("[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results.")
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = [depth_filenames[i] for i in indices]
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32))
        scene_box.aabb += scene_box.aabb[1, :] * torch.Tensor(self.config.offset)[None, :]

        img_0 = imageio.v2.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
        cx = image_width / 2.0
        cy = image_height / 2.0

        cameras = Cameras(
            camera_to_worlds=poses[:, :3, :4],
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        self.max_fid = max([int(Path(frame["file_path"]).stem.split("_")[-1]) for frame in meta["frames"]])
        fids = [Path(name).stem.split("_")[-1] for name in image_filenames]
        ids = torch.tensor([int(fid) / self.max_fid for fid in fids], dtype=torch.float32)[:, None]
        self.num_atrbs, atrb_vals, atrb_val_masks, atrb_masks, mask_valids = read_attributes(fids, data_dir, self.config.norm_vals, True)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames,
                "atrb_vals": atrb_vals,
                "atrb_val_masks": atrb_val_masks,
                "atrb_masks": atrb_masks,
                "mask_valids": mask_valids,
                "ids": ids,
            },
        )
        return dataparser_outputs

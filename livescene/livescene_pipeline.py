import typing
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Literal, Optional, Type

import torch
import torch.distributed as dist
import torchvision.utils as vutils
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from livescene.datamanager.livescene_datamanager import LiveSceneDataManager
from livescene.livescene_controller import LiveSceneController, LiveSceneControllerConfig
from livescene.livescene_model import LiveSceneModel


@dataclass
class LiveScenePipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: LiveScenePipeline)
    """target class to instantiate"""
    controller: LiveSceneControllerConfig = LiveSceneControllerConfig()
    """viewer controller"""
    use_gt_val: bool = False
    """use ground truth values for attribute values"""
    train_with_lang: bool = False
    """train with language features"""


class LiveScenePipeline(VanillaPipeline):
    def __init__(
        self,
        config: LiveScenePipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super(VanillaPipeline, self).__init__()
        self.config = config
        self.test_mode = test_mode
        self.controller: LiveSceneController = config.controller.setup(train_with_lang=self.config.train_with_lang)
        self.datamanager: LiveSceneDataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, image_encoder=self.controller.clip_encoder, use_gt_val=self.config.use_gt_val, train_with_lang=self.config.train_with_lang
        )
        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        num_atrbs = self.datamanager.dataparser.num_atrbs
        self.controller.register_slider(num_attributes=num_atrbs)

        for i in range(len(config.model.proposal_lang_net_args_list)):
            config.model.proposal_lang_net_args_list[i]["ncls"] = num_atrbs

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            viewer_controller=self.controller,
            use_gt_val=self.config.use_gt_val,
            train_with_lang=self.config.train_with_lang,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(LiveSceneModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

    @profiler.time_function
    def get_average_image_metrics(
        self,
        data_loader,
        image_prefix: str,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        """Iterate over all the images in the dataset and get the average.

        Args:
            data_loader: the data loader to iterate over
            image_prefix: prefix to use for the saved image filenames
            step: current training step
            output_path: optional path to save rendered images to
            get_std: Set True if you want to return std with the mean metric.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(data_loader)
        if output_path is not None:
            output_path.mkdir(exist_ok=True, parents=True)
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            idx = 0
            for camera, batch in data_loader:
                # time this the following line
                inner_start = time()
                image_idx = batch["image_idx"]

                camera.metadata = {
                    "id": self.datamanager.eval_dataset.metadata["ids"][image_idx: image_idx+1].to(self.device),
                }
                if self.config.use_gt_val:
                    camera.metadata["atrb_val"] = self.datamanager.eval_dataset.metadata["atrb_vals"][image_idx: image_idx+1].to(self.device)

                outputs = self.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = self.model.get_image_metrics_and_images(outputs, batch)
                if output_path is not None:
                    for key in image_dict.keys():
                        image = image_dict[key]  # [H, W, C] order
                        vutils.save_image(
                            image.permute(2, 0, 1).cpu(), output_path / f"{image_prefix}_{key}_{idx:04d}.png"
                        )

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = (num_rays / (time() - inner_start)).item()
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = (metrics_dict["num_rays_per_sec"] / (height * width)).item()
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
                idx = idx + 1

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        for key, value in metrics_dict.items():
            print(f"{key}: {value}")

        self.train()
        return metrics_dict

import re
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Type

import clip
import open_clip
import torch
import torchvision
from nerfstudio.configs import base_config as cfg
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.viewer.server.viewer_elements import ViewerSlider, ViewerText
from omegaconf import OmegaConf
from torch import nn


@dataclass
class LiveSceneControllerConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: LiveSceneController)

    clip_type: str = "openclip"
    clip_args: Dict[str, str] = to_immutable_dict(
        {
            "clip_model_type": "ViT-B/16",
            "clip_n_dims": 512,
        }
    )
    openclip_args: Dict[str, str] = to_immutable_dict(
        {
            "clip_model_type": "ViT-B-16",
            "clip_model_pretrained": "laion2b_s34b_b88k",
            "clip_n_dims": 512,
        }
    )
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")

class LiveSceneController(nn.Module):
    def __init__(self, config: LiveSceneControllerConfig, **kwargs):
        super().__init__()
        self.train_with_lang = kwargs["train_with_lang"]
        self.config = config
        self.register_input()
        self.clip_encoder = LiveSceneClipEncoder(config)

    def register_slider(self, num_attributes: int):
        def register_individual_slider(id):
            def gui_cb(element: ViewerSlider):
                print(element.value)

            setattr(self, f"slider_{id}", ViewerSlider(name=f"attribute_{id}", default_value=0.0, min_value=0.0, max_value=1.0, step=0.01, cb_hook=gui_cb))

        self.num_attributes = num_attributes
        for i in range(num_attributes):
            register_individual_slider(i)

    def register_input(self):
        def gui_cb(element: ViewerText):
            print(element.value)

        if self.train_with_lang:
            setattr(self, "prompt", ViewerText(name="prompt", default_value="", cb_hook=gui_cb))

    def get_slider_vals(self):
        attrb_vals = []
        for i in range(self.num_attributes):
            attrb_vals.append(float(getattr(self, f"slider_{i}").value))
        return attrb_vals

    def get_obj_status(self):
        template = r'^(open|close) (.*)'
        match = re.match(template, self.prompt.value)
        if match:
            action, obj = match.groups()
            return obj, 1 if action == "open" else 0
        else:
            return None, None

    def get_atrb_vals(self, field_embed: torch.Tensor):
        atrb_val = torch.tensor(self.get_slider_vals(), device=field_embed.device)[None, :]
        obj, target_val = self.get_obj_status()
        if obj is not None:
            probs = self.clip_encoder.get_relevancy(field_embed, obj)
            pos_prob = probs[..., 0].reshape(-1, self.num_attributes)
            pos_prob = pos_prob[torch.any(pos_prob >= 0.5, dim=-1)]
            if pos_prob.shape[0] >= 10:
                targetid = torch.bincount(pos_prob.max(-1)[1]).max(-1)[1]
                atrb_val[:, targetid] = target_val
        return atrb_val

# Adapted from lerf.lerf.lerf.LERFModel
class LiveSceneClipEncoder(ABC):
    def __init__(self, config):
        self.config = config
        self.config.args = OmegaConf.create()
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        if self.config.clip_type == "clip":
            self.config.args.update(self.config.clip_args)
            model, _ = clip.load(self.config.args.clip_model_type)
            self.tokenizer = clip.tokenize
        if self.config.clip_type == "openclip":
            self.config.args.update(self.config.openclip_args)
            model, _, _ = open_clip.create_model_and_transforms(
                self.config.args.clip_model_type,
                pretrained=self.config.args.clip_model_pretrained,
                precision="fp16",
            )
            self.tokenizer = open_clip.get_tokenizer(self.config.args.clip_model_type)
        model.eval()
        self.model = model.to("cuda")
        for param in self.model.parameters():
            param.requires_grad = False
        self.clip_n_dims = self.config.args.clip_n_dims

        self.negatives = self.config.negatives
        tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
        self.neg_embeds = self.model.encode_text(tok_phrases)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    @property
    def name(self) -> str:
        if self.config.clip_type == "clip":
            return "clip_{}".format(self.config.args.clip_model_type)
        if Path(self.config.args.clip_model_pretrained).name == str(Path(self.config.args.clip_model_pretrained)):
            return "openclip_{}_{}".format(self.config.args.clip_model_type, self.config.args.clip_model_pretrained)
        else:
            return "openclip_{}".format(Path(self.config.args.clip_model_pretrained).parent.name)

    @property
    def embedding_dim(self) -> int:
        return self.clip_n_dims

    def process_image(self, input):
        return self.process(input.to("cuda")).half()

    def encode_image(self, input):
        return self.model.encode_image(input.to("cuda"))

    def encode_text(self, input):
        token = self.tokenizer(input).to("cuda")
        text_embed = self.model.encode_text(token)
        return text_embed

    def get_phrases_embed(self, obj) -> torch.Tensor:
        self.pos_embeds = self.encode_text(obj)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        return phrases_embeds

    def get_relevancy(self, field_embed: torch.Tensor, obj: str) -> torch.Tensor:
        phrases_embeds = self.get_phrases_embed(obj)
        field_embed = (field_embed / field_embed.norm(dim=-1, keepdim=True)).half()
        output = torch.mm(field_embed, phrases_embeds.to(phrases_embeds.dtype).T)  # rays x phrases
        positive_vals = output[..., 0:1]
        negative_vals = output[..., 1:]
        repeated_pos = positive_vals.repeat(1, negative_vals.shape[-1])

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], negative_vals.shape[-1], 2))[:, 0, :]

import json
from pathlib import Path
from typing import Dict

import cv2 as cv
import numpy as np
import rasterio.features
import torch
import yaml
from omegaconf import OmegaConf
from shapely.geometry import Polygon


def read_attributes(fids, data_dir, norm_vals, fewshot):
    def load_from_yaml(file: Path):
        with open(file, "r") as f:
            return yaml.safe_load(f)

    id2cls = load_from_yaml(data_dir / "mapping.yaml")
    cls2id = {v: k for k, v in id2cls.items()}
    num_atrbs = len(cls2id)

    atrb_vals, atrb_val_masks = interp_atrb_values(data_dir / "key_frame_value.yaml", fids, num_attributes=num_atrbs, cls_to_id_mapping=cls2id, norm_vals=norm_vals, fewshot=fewshot)
    atrb_masks = [np.load(data_dir / "masks" / f"{fid}.npy") for fid in fids]
    H, W, _ = atrb_masks[0].shape
    mask_valids = (np.stack(atrb_masks).sum(axis=(1, 2)) == 0) | (np.stack(atrb_masks).sum(axis=(1, 2)) > H * W / 300)

    atrb_vals = torch.from_numpy(np.stack(atrb_vals)).float()  # (N_images, num_atrb+1)
    atrb_val_masks = torch.from_numpy(np.stack(atrb_val_masks)).bool()  # (N_images, num_atrb+1)
    atrb_masks = torch.from_numpy(np.stack(atrb_masks)).bool().float()  # (N_images, num_atrb+1)
    mask_valids = torch.from_numpy(mask_valids).bool()  # (N_images, num_atrb+1)
    return num_atrbs, atrb_vals, atrb_val_masks, atrb_masks, mask_valids


def load_conerf_annotation(dir_path: Path, fids, height: int, width: int, scale: float, num_attributes: int, cls_to_id_mapping: Dict[str, int]):
    """
    * load annotation from json file, and create a annotation mask with shape (h, w, num + 1)
    ret:
    mask_labels (H, W, num + 1), where num is the number of attributes
    valids (num + 1), the valids of mask
    """
    atrb_masks, mask_valids = [], []
    for fid in fids:
        mask_labels = np.zeros((height, width, num_attributes + 1), dtype=np.bool_)
        path = dir_path / f"{fid}.json"
        if not path.exists() or num_attributes == 0:
            valids = np.zeros(1, dtype=np.bool_)
            mask_labels[..., -1] = True
        else:
            with open(path) as f:
                data = json.load(f)

            for datum in data["shapes"]:
                polygon = Polygon(np.array(datum["points"]) / scale)
                cls_id = cls_to_id_mapping[datum["label"]]
                mask = rasterio.features.rasterize([polygon], out_shape=(height, width))
                mask_labels[..., cls_id] = mask_labels[..., cls_id] | mask

            # * add background class
            mask_labels[mask_labels.sum(axis=-1) == 0, -1] = True
            valids = np.ones(1, dtype=np.bool_)

        atrb_masks.append(mask_labels)
        mask_valids.append(valids)
    return atrb_masks, mask_valids


def load_coco_annotations(path, fids, height: int, width: int, scale: float = 1.0, num_attributes: int = 1):
    """
    * load annotation from json file, and create a annotation mask with shape (h, w, num + 1)
    ret: mask_labels (H, W, num + 1), where num is the number of attributes
    valids (num + 1), the valids of mask
    """
    # load annotation json
    with open(path) as f:
        data = json.load(f)

    # image id to fid
    image_id_to_fid = {}
    for img_dict in data["images"]:
        fid = img_dict["file_name"].split("_")[0]
        image_id_to_fid[img_dict["id"]] = fid

    # annotations
    fid_to_annotations = {}
    for annotations in data["annotations"]:
        image_id = annotations["image_id"]
        fid = image_id_to_fid[image_id]
        if fid not in fid_to_annotations:
            fid_to_annotations[fid] = []
        fid_to_annotations[fid].append({"category_id": annotations["category_id"], "points": np.array(annotations["segmentation"]).reshape(-1, 2)})

    # load annotation Polygon
    atrb_masks, mask_valids = [], []
    for fid in fids:
        mask_labels = np.zeros((height, width, num_attributes + 1), dtype=np.bool_)

        if fid not in fid_to_annotations or num_attributes == 0:
            # * w/o annotation should return a mask with all 0
            valids = np.zeros(1, dtype=np.bool_)
        else:
            for annotations in fid_to_annotations[fid]:
                polygon = Polygon(annotations["points"] / scale)
                cls_id = annotations["category_id"]
                mask = rasterio.features.rasterize([polygon], out_shape=(height, width))
                mask_labels[..., cls_id] = mask_labels[..., cls_id] | mask

            # * add background class
            mask_labels[mask_labels.sum(axis=-1) == 0, -1] = True
            valids = np.ones(1, dtype=np.bool_)

        atrb_masks.append(mask_labels)
        mask_valids.append(valids)

    return atrb_masks, mask_valids


def load_blender_annotations(path: Path, fids, height: int, width: int, num_attributes: int = 1):
    """
    * load annotation from json file, and create a annotation mask with shape (h, w, num + 1)
    ret: mask_labels (H, W, num + 1), where num is the number of attributes
    valids (num + 1), the valids of mask
    """
    atrb_masks, mask_valids = [], []
    for fid in fids:
        mask_labels = np.zeros((height, width, num_attributes + 1), dtype=np.bool_)
        seg_path = path / f"{fid}_segmentation.npy"
        if not seg_path.exists() or num_attributes == 0:
            # * w/o annotation should return a mask with all 0
            valids = np.zeros(1, dtype=np.bool_)
        else:
            mask_labels[..., :num_attributes] = np.load(seg_path)[..., :num_attributes]
            # * add background class
            mask_labels[mask_labels.sum(axis=-1) == 0, -1] = 1
            valids = np.ones(1, dtype=np.bool_)

        atrb_masks.append(mask_labels)
        mask_valids.append(valids)

    return atrb_masks, mask_valids


def load_conerf_values(path: Path, fids, num_attributes, norm_vals=True):
    annotations_in_file = OmegaConf.load(path)
    fid_to_id_mapping = {int(fid): i for i, fid in enumerate(fids)}
    atrb_vals = np.zeros((len(fids), num_attributes), dtype=np.float32)
    atrb_val_masks = np.zeros((len(fids), num_attributes + 1), dtype=np.float32)
    atrb_val_masks[..., -1] = True

    for entry in annotations_in_file:
        fid, cls = entry["frame"], entry["class"]
        if fid in fid_to_id_mapping:
            atrb_vals[fid_to_id_mapping[fid]][cls] = entry["value"]
            atrb_val_masks[fid_to_id_mapping[fid]][cls] = True
    # if norm_vals:
    #     atrb_vals = (atrb_vals - atrb_vals.min(axis=0, keepdims=True)) / (atrb_vals.max(axis=0, keepdims=True) - atrb_vals.min(axis=0, keepdims=True))
    atrb_vals = 0.5 * (atrb_vals + 1)
    atrb_vals = np.hstack([np.zeros((atrb_vals.shape[0], 1)), atrb_vals])

    return atrb_vals, atrb_val_masks


def load_prompts(split, path, fids, cls_to_id_mapping):
    data = OmegaConf.load(path)
    data = data["val" if split == "test" else split]

    fid_to_id_mapping = {int(fid): i for i, fid in enumerate(fids)}

    prompts = np.zeros((len(cls_to_id_mapping), 2), dtype=object)

    for prompt, sfid in data.items():
        fid = int(sfid.split("_")[-1])
        if fid in fid_to_id_mapping:
            prompts[cls_to_id_mapping[prompt], 0] = prompt
            prompts[cls_to_id_mapping[prompt], 1] = fid_to_id_mapping[fid]
    return prompts


def interp_atrb_values(path, fids, cls_to_id_mapping, num_attributes, norm_vals=True, fewshot=False):
    data = OmegaConf.load(path)
    fid_to_id_mapping = {int(fid): i for i, fid in enumerate(fids)}
    xids_2d, vals_2d = [[] for _ in range(num_attributes)], [[] for _ in range(num_attributes)]

    fewshot_values = np.zeros((len(fids), num_attributes), dtype=np.float32)
    atrb_val_masks = np.zeros((len(fids), num_attributes + 1), dtype=np.float32)
    atrb_val_masks[..., -1] = True

    for sfid, item in data.items():
        fid = int(sfid.split("_")[-1])
        for cls, v in item.items():
            xids_2d[cls_to_id_mapping[cls]].append(fid)
            vals_2d[cls_to_id_mapping[cls]].append(v)
            if fid in fid_to_id_mapping:
                fewshot_values[fid_to_id_mapping[fid]][cls_to_id_mapping[cls]] = v
                atrb_val_masks[fid_to_id_mapping[fid]][cls_to_id_mapping[cls]] = True

    if fewshot:
        if norm_vals:
            fewshot_values = (fewshot_values - fewshot_values.min(axis=0, keepdims=True)) / (fewshot_values.max(axis=0, keepdims=True) - fewshot_values.min(axis=0, keepdims=True))
        fewshot_values = np.hstack([fewshot_values, np.zeros((fewshot_values.shape[0], 1))])
        return fewshot_values, atrb_val_masks

    # * interp
    xfids = [int(fid) for fid in fids]
    interp_values = []
    for i in range(num_attributes):
        interp_values.append(np.interp(xfids, xids_2d[i], vals_2d[i]))
    interp_values = np.array(interp_values).transpose(1, 0)  # (n, num_attributes)
    if norm_vals:
        interp_values = (interp_values - interp_values.min(axis=0, keepdims=True)) / (interp_values.max(axis=0, keepdims=True) - interp_values.min(axis=0, keepdims=True))

    # (n, num_attributes) - > (n, num_attributes + 1) fill with 0
    inter_atrb_values = np.hstack([interp_values, np.zeros((interp_values.shape[0], 1))])
    atrb_val_masks = np.ones_like(inter_atrb_values, dtype=np.bool_)
    return inter_atrb_values, atrb_val_masks


def load_comb_annotations(path, fids, height: int, width: int, scale: float = 1.0, num_attributes: int = 1) -> np.ndarray:
    """
    * load annotation from json file, and create a annotation mask with shape (h, w, num + 1)
    ret: mask_labels (H, W, num + 1), where num is the number of attributes
    valids (num + 1), the valids of mask
    """
    # load annotation Polygon
    atrb_masks = []
    for fid in fids:
        file_path = path / f"{fid}.npy"
        if not file_path.exists() or num_attributes == 0:
            # * w/o annotation should return a mask with all 0
            mask_labels = np.zeros((height, width, num_attributes + 1), dtype=np.bool_)
            valids = np.zeros(1, dtype=np.bool_)
        else:
            mask_labels = np.load(file_path)  # (h, w, c)
            valids = np.ones(1, dtype=np.bool_)

            # resize the mask_labels
            if scale != 1.0:
                mask_labels = cv.resize(mask_labels, (mask_labels.shape[1] // scale, mask_labels.shape[0] // scale), interpolation=cv.INTER_NEAREST)

        atrb_masks.append(mask_labels)
    return atrb_masks, valids

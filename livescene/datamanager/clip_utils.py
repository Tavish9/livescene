import cv2
import numpy as np
import torch


def embed_clip(model, images, masks, mask_valids):
    clip_process = torch.zeros(len(images), masks.shape[-1], 3, 224, 224, dtype=torch.float16)
    for i in range(len(images)):
        image = images[i]
        mask = masks[i]
        for j in range(masks.shape[-1]):
            if not mask_valids[i, j] or mask[..., j].sum() < 500:
                continue
            individual_mask = mask[..., j]
            cur_image = image.clone()
            cur_image[:, ~individual_mask] = 0
            x, y, w, h = cv2.boundingRect(individual_mask.astype(np.uint8))
            cropped_image = cur_image[:, y : y + h, x : x + w]
            processed_image = model.process_image(cropped_image[None, ...])
            clip_process[i, j] = processed_image
    clip_embeddings = model.encode_image(clip_process.flatten(0, 1)).unflatten(0, (len(images), masks.shape[-1])) # (N, C+1， 512)
    clip_embeddings = clip_embeddings / clip_embeddings.norm(dim=-1, keepdim=True)
    return torch.from_numpy(masks), clip_embeddings.cpu()


def embed_clip_avg_views(model, images, masks, mask_valids):
    processed_images = {i: [] for i in range(masks.shape[-1])}
    for i in range(len(images)):
        image = images[i]
        mask = masks[i]
        for j in range(masks.shape[-1]):
            if not mask_valids[i, j] or mask[..., j].sum() < 500:
                continue
            individual_mask = mask[..., j]
            cur_image = image.clone()
            cur_image[:, ~individual_mask] = 0
            x, y, w, h = cv2.boundingRect(individual_mask.astype(np.uint8))
            cropped_image = cur_image[:, y : y + h, x : x + w]
            processed_image = model.process_image(cropped_image)
            processed_images[j].append(processed_image)
        image[:, mask[..., -1] == 0] = 0
        processed_image = model.process_image(image)
        processed_images[masks.shape[-1]-1].append(processed_image)
    processed_images = {id: torch.stack(image_list) for id, image_list in processed_images.items()}
    clip_embeddings = {id: model.encode_image(image_list).sum(0) for id, image_list in processed_images.items()}
    clip_embeddings = torch.stack([(clip_embeddings[i] / clip_embeddings[i].norm(dim=-1, keepdim=True)).cpu() for i in range(masks.shape[-1])])  # (C+1， 512)
    return torch.from_numpy(masks), clip_embeddings.repeat(len(images), 1, 1)

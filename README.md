# <img src="https://github.com/Tavish9/livescene/raw/main/docs/static/image/icons/favicon.webp" height="30px"> LiveScene: Language Embedding Interactive Radiance Fields for Physical Scene Rendering and Control
This is the official implementation for [LiveScene](https://livescenes.github.io/).

> LiveScene: Language Embedding Interactive Radiance Fields for Physical Scene Rendering and Control <br />
> [Delin Qu*](https://delinqu.github.io), [Qizhi Chen*](https://github.com/Tavish9), Pingrui Zhang Xianqiang Gao, Binzhao, Zhigang Wang, Dong Wang†, [Xuelong Li†](https://scholar.google.com/citations?user=ahUibskAAAAJ)

### [Paper](https://arxiv.org/abs/2406.16038) | [Demo](https://livescenes.github.io) | [Video](https://youtu.be/4gqUoCFK_0Q?si=fPY1HHYtUU6D6lkf)

<div align='center'>
<img src="https://github.com/Tavish9/livescene/raw/main/docs/static/image/pipeline.png" >
</div>

## Update
- [x] Project Pages for LiveScene: Language Embedding Interactive Radiance Fields for Physical Scene Rendering and Control [2024-5-18]
- [x] DATASET for LiveScene [2024-10-21]
- [x] Video and slides [2024-11-04]
- [x] Code for LiveScene [2025-01-09]


## ⚙️ Installation
### 1. Install Nerfstudio
Follow the official instructions to install the latest version of [nerfstudio](https://docs.nerf.studio/quickstart/installation.html).
### 2. Install Livescene
Clone the repository and install the package:
```
git clone https://github.com/Tavish9/livescene.git
pip install -e .
```

## 📥 Download Dataset
The `InterReal` and `OmniSim` datasets are available on [Huggingface](https://huggingface.co/datasets/IPEC-COMMUNITY/LiveScene). 

To download the entire dataset using `huggingface_cli`:
```
huggingface-cli download --local-dir livescene_dataset --repo-type dataset IPEC-COMMUNITY/LiveScene
```

## 🚀 Running LiveScene
### 1. Check running options
To view available options for training:
```
ns-train livescene --help
```
use `livescene-real-data` for `InterReal` dataset and `livescene-sim-data` for `OmniSim` dataset.

### 2. Launch Training
You can use it like any other third-party nerfstudio project.
```
ns-train livescene --data /path/to/your/livescene_dataset/scene_name livescene-real/sim-data
```

### 3. Interact with viewer
**Slider Interact**

Use slider to control specific objects within scene.
<div align="center">
  <img src="https://github.com/user-attachments/assets/c31dbd23-5ece-429e-86d6-79be8381fb8f" style="width: 45%;">
  <img src="https://github.com/user-attachments/assets/4125a17f-4cdb-47dc-8a6a-8739037e8828" style="width: 45%;">
</div>

**Language Interact**

Use natural language to interact with the 3D scenes.
<div align="center">
  <img src="https://github.com/user-attachments/assets/b6b8a6a8-33ac-46a0-a915-6ebc00172c72" style="width: 45%;">
  <img src="https://github.com/user-attachments/assets/dd9c053e-a336-47ed-a861-77329f4bd9b2" style="width: 45%;">
</div>


**Relevancy Map**

Visualize relevancy map in the scene.
<div align="center">
  <img src="https://github.com/user-attachments/assets/9def3038-1568-4c46-9c25-7074e2f46bd9" style="width: 30%;">
  <img src="https://github.com/user-attachments/assets/688673dc-d012-4a03-a58c-331f04cf4dd9" style="width: 30%;">
  <img src="https://github.com/user-attachments/assets/adb713f0-2c70-45e4-bec8-26408f2e4639" style="width: 30%;">
</div>

## Acknowledgement

We adapt codes from some awesome repositories, including [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [Omnigibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html), [Kplanes_nerfstudio](https://github.com/Giodiro/kplanes_nerfstudio), [Lerf](https://github.com/kerrj/lerf/), and [Conerf](https://github.com/kacperkan/conerf). Thanks for making the code available! 🤗

## Citation

If you use this work or find it helpful, please consider citing: (bibtex)
```
@misc{livescene2024,
  title={LiveScene: Language Embedding Interactive Radiance Fields for Physical Scene Rendering and Control}, 
  author={Delin Qu, Qizhi Chen, Pingrui Zhang, Xianqiang Gao, Bin Zhao, Zhigang Wang, Dong Wang†, Xuelong Li†},
  year={2024},
  eprint={2406.16038},
  archivePrefix={arXiv},
}
```

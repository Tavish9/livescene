# <img src="https://github.com/livescenes/livescenes.github.io/raw/main/docs/static/image/icons/favicon.webp" height="30px"> LiveScene: Language Embedding Interactive Radiance Fields for Physical Scene Rendering and Control
This is the official implementation for [LiveScene](https://livescenes.github.io/).

### [Paper](https://arxiv.org/abs/2406.16038) | [Demo](https://livescenes.github.io) | [Video](https://youtu.be/4gqUoCFK_0Q?si=fPY1HHYtUU6D6lkf)


<div align='center'>
<img src="https://github.com/livescenes/livescenes.github.io/raw/main/docs/static/image/pipeline.png" >
</div>

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
  <img src="https://github.com/user-attachments/assets/0a163ecf-771b-43b2-8f96-b3116a879389" style="width: 45%;">
  <img src="https://github.com/user-attachments/assets/46b577a2-ea7b-42b2-b3e8-e0ad627f771c" style="width: 45%;">
</div>

**Language Interact**

Use natural language to interact with the 3D scenes.
<div align="center">
  <img src="https://github.com/user-attachments/assets/0a163ecf-771b-43b2-8f96-b3116a879389" style="width: 45%;">
  <img src="https://github.com/user-attachments/assets/e0729bb1-1900-44ef-92dc-2d109ba5f416" style="width: 45%;">
</div>


**Relevancy Map**

Visualize relevancy map in the scene.
<div align="center">
  <img src="https://github.com/user-attachments/assets/4ad92324-c1aa-46c7-a192-d538b8de97a8" style="width: 30%;">
  <img src="https://github.com/user-attachments/assets/6eeb23ac-9eb9-402f-b285-84991c9cdcb7" style="width: 30%;">
  <img src="https://github.com/user-attachments/assets/a549eb33-64df-474f-a232-ed03f1a0ca8f" style="width: 30%;">
</div>

## 🙏 Acknowledgement

We adapt codes from several awesome repositories, including [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio), [Omnigibson](https://behavior.stanford.edu/omnigibson/getting_started/installation.html), [Kplanes_nerfstudio](https://github.com/Giodiro/kplanes_nerfstudio), [Lerf](https://github.com/kerrj/lerf/), and [Conerf](https://github.com/kacperkan/conerf). Thank you all for making the code available! 🤗


## 📚 Citation
If you use this work or find it helpful, please consider citing: (bibtex)
```
@misc{livescene2024,
  title={LiveScene: Language Embedding Interactive Radiance Fields for Physical Scene Rendering and Control}, 
  author={Delin Qu*, Qizhi Chen*, Pingrui Zhang, Xianqiang Gao, Bin Zhao, Zhigang Wang, Dong Wang†, Xuelong Li†},
  year={2024},
  eprint={2406.16038},
  archivePrefix={arXiv},
}
```

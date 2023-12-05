## [CVPR 23] Explicit Visual Prompting for Low-Level Structure Segmentations
 
Weihuang Liu, [Xi Shen](https://xishen0220.github.io/), [Chi-Man Pun](https://www.cis.um.edu.mo/~cmpun/), and [Xiaodong Cun](https://vinthony.github.io/) by University of Macau and Tencent AI Lab.

<a href='https://arxiv.org/abs/2303.10883'><img src='https://img.shields.io/badge/ArXiv-2303.08524-red' /></a> 
  <a href='https://nifangbaage.github.io/Explicit-Visual-Prompt/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
 

## 🔥 News
The extension work of [Explicit Visual Prompting for Universal Foreground Segmentations](https://arxiv.org/abs/2305.18476) is released. The codes, models, and results can be found in this repository.

## Abstract
> We consider the generic problem of detecting low-level structures in images, which includes segmenting the manipulated parts, identifying out-of-focus pixels, separating shadow regions, and detecting concealed objects.
Whereas each such topic has been typically addressed with a domain-specific solution, we show that a unified approach performs well across all of them. 
We take inspiration from the widely-used pre-training and then prompt tuning protocols in NLP and propose a new visual prompting model, named Explicit Visual Prompting (EVP). 
Different from the previous visual prompting which is typically a dataset-level implicit embedding, our key insight is to enforce the tunable parameters focusing on the explicit visual content from each individual image, i.e., the features from frozen patch embeddings and the input's high-frequency components.
The proposed EVP significantly outperforms other parameter-efficient tuning protocols under the same amount of tunable parameters (5.7% extra trainable parameters of each task). EVP also achieves state-of-the-art performances on diverse low-level structure segmentation tasks compared to task-specific solutions. 


## Overview
<p align="center">
  <img width="80%" alt="teaser" src="teaser/teaser.png">
</p>
We propose a unified method for four low-level structure segmentation tasks: camouflaged object, forgery, shadow and defocus blur detection (left). 
Our approach relies on a pre-trained frozen transformer backbone that leverages explicit extracted features, e.g., the frozen embedded features and high-frequency components, to prompt knowledge (right).


## Pipeline
<p align="center">
  <img width="40%" alt="pipeline" src="teaser/pipeline.png">
</p>
We remodulate the features via the Embedding Tune and the HFC Tune. The Adaptor is designed to efficiently obtain prompts. 

## Environment
This code was implemented with Python 3.6 and PyTorch 1.8.1. You can install all the requirements via:
```bash
pip install -r requirements.txt
```

## Demo
```bash
python demo.py --input [INPUT_PATH] --model [MODEL_PATH] --prompt [PROMPT_PATH] --resolution [HEIGHT],[WIDTH] --config [CONFIG_PATH]
```
`[INPUT_PATH]`: input image

`[PROMPT_PATH]`: prompt checkpoint

`[MODEL_PATH]`: backbone checkpoint

`[HEIGHT]`: target height

`[WIDTH]`: target width

`[CONFIG_PATH]`: config file


## Quick Start
1. Download the dataset and put it in ./load.
2. Download the pre-trained SegFormer backbone.
3. Training:
```bash
python train.py --config configs/train/segformer/train_segformer_evp_defocus.yaml 
```
4. Evaluation:
```bash
python test.py --config configs/test/test_defocus.yaml  --model mit_b4.pth --prompt ./save/_train_segformer_evp_defocus/prompt_epoch_last.pth
```
5. Visualization:
```bash
python demo.py --input defocus.png --model ./mit_b4.pth --prompt ./save/_train_segformer_evp_defocus/prompt_epoch_last.pth --resolution 320,320 --config configs/demo.yaml
```

## Train
```bash
python train.py --config [CONFIG_PATH]
```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH] --prompt [PROMPT_PATH]
```

## Models and Results

Please find the pre-trained models and results [here](https://drive.google.com/drive/folders/1dt24hFhLTa-C4gdbZbGo0kdZA_OYaDMS?usp=sharing).


## Dataset

### Camouflaged Object Detection
- **COD10K**: https://github.com/DengPingFan/SINet/
- **CAMO**: https://drive.google.com/open?id=1h-OqZdwkuPhBvGcVAwmh0f1NGqlH_4B6
- **CHAMELEON**: https://www.polsl.pl/rau6/datasets/

### Defocus Blur Detection
- **DUT**: http://ice.dlut.edu.cn/ZhaoWenda/BTBCRLNet.html
- **CUHK**: http://www.cse.cuhk.edu.hk/leojia/projects/dblurdetect/

### Forgery Detection
- **CAISA**: https://github.com/namtpham/casia2groundtruth
- **IMD2020**: http://staff.utia.cas.cz/novozada/db/

### Shadow Detection
- **ISTD**: https://github.com/DeepInsight-PCALab/ST-CGAN
- **SBU**: https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html


## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{liu2023explicit,
  title={Explicit visual prompting for low-level structure segmentations},
  author={Liu, Weihuang and Shen, Xi and Pun, Chi-Man and Cun, Xiaodong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19434--19445},
  year={2023}
}
```

## Acknowledgements

EVP code borrows heavily from [LIIF](https://github.com/yinboc/liif), [SETR](https://github.com/fudan-zvg/SETR) and [SegFormer](https://github.com/NVlabs/SegFormer). We thank the author for sharing their wonderful code. 

# IntagHand

This repository contains a pytorch implementation of "[Interacting Attention Graph for Single Image Two-Hand Reconstruction](http://www.liuyebin.com/IntagHand/Intaghand.html)". 

Mengcheng Li, [Liang An](https://anl13.github.io), [Hongwen Zhang](https://hongwenzhang.github.io), Lianpeng Wu, Feng Chen, [Tao Yu](http://ytrock.com/), [Yebin Liu](http://www.liuyebin.com/)

Tsinghua University & Hisense Inc.

CVPR 2022 (Oral)



**2023.02.02 Update: add an example of training code**



![pic](http://www.liuyebin.com/IntagHand/assets/results2.png)

## Requirements

- Tested with python3.7 on Ubuntu 16.04, CUDA 10.2.

### packages

- pytorch (tested on 1.10.0+cu102)

- torchvision (tested on 0.11.0+cu102)

- pytorch3d (tested on 0.6.1)

- numpy

- OpenCV

- tqdm

- yacs >= 0.1.8

### Pre-trained model and data

- Download  necessary assets (including the pre-trained models) from [misc.tar.gz](https://github.com/Dw1010/IntagHand/releases/download/v0.0/misc.tar.gz) and unzip it.
- Register and download [MANO](https://mano.is.tue.mpg.de/)  data. Put `MANO_LEFT.pkl` and `MANO_RIGHT.pkl` in `misc/mano`

After collecting the above necessary files, the directory structure of `./misc` is expected as follows:

```
./misc
├── mano
│   └── MANO_LEFT.pkl
│   └── MANO_RIGHT.pkl
├── model
│   └── config.yaml
│   └── interhand.pth
│   └── wild_demo.pth
├── graph_left.pkl
├── graph_right.pkl
├── upsample.pkl
├── v_color.pkl

```

## DEMO

1. Real-time demo :

```
python apps/demo.py --live_demo
```
2. Single-image reconstruction  :

```
python apps/demo.py --img_path demo/ --save_path demo/
```
Results will be stored in folder `./demo`

**Noted**: We don't operate hand detection, so hands are expected to be roughly at the center of image and take approximately 70-90% of the image area.

## Training

1. Download [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset and unzip it. (**Noted**: we used the `v1.0_5fps` version and `H+M` subset for training and evaluating)

2. Process the dataset by :
```
python dataset/interhand.py --data_path PATH_OF_INTERHAND2.6M --save_path ./interhand2.6m/
```
Replace `PATH_OF_INTERHAND2.6M` with your own store path of [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset. 

3. Try the training code:
```
python apps/train.py utils/defaults.yaml
```

The output model and TensorBoard log file would be store in `./output`.
If you have multiple GPUs on your device, set `--gpu` to use them.  For example, use:

```
python apps/train.py utils/defaults.yaml --gpu 0,1,2,3
```
to train model on 4 GPUs.

4. We highly recommend you to try different loss weight and fine-turn the model with lower learning rate to get better result. The training configuration can be modified in `utils/defaults.yaml`.

## Evaluation

1. Download [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset and unzip it. (**Noted**: we used the `v1.0_5fps` version and `H+M` subset for training and evaluating)

2. Process the dataset by :
```
python dataset/interhand.py --data_path PATH_OF_INTERHAND2.6M --save_path ./interhand2.6m/
```
Replace `PATH_OF_INTERHAND2.6M` with your own store path of [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset. 

3. Run evaluation:
```
python apps/eval_interhand.py --data_path ./interhand2.6m/
```

You would get following output :

```
joint mean error:
    left: 8.93425289541483 mm, right: 8.663229644298553 mm
    all: 8.798741269856691 mm
vert mean error:
    left: 9.173248894512653 mm, right: 8.890160359442234 mm
    all: 9.031704626977444 mm
```


## Acknowledgement

The pytorch implementation of MANO is based on [manopth](https://github.com/hassony2/manopth). The GCN network is based on [hand-graph-cnn](https://github.com/3d-hand-shape/hand-graph-cnn). The heatmap generation and inference is based on [DarkPose](https://github.com/ilovepose/DarkPose). We thank the authors for their great job!

## Citation

If you find the code useful in your research, please consider citing the paper.

```
@inproceedings{Li2022intaghand,
title={Interacting Attention Graph for Single Image Two-Hand Reconstruction},
author={Li, Mengcheng and An, Liang and Zhang, Hongwen and Wu, Lianpeng and Chen, Feng and Yu, Tao and Liu, Yebin},
booktitle={IEEE/CVF Conf. on Computer Vision and Pattern Recognition (CVPR)},
month=jun,
year={2022},
}
```


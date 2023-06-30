# *SimpModeling*

This repository includes the prototype system of *SimpModeling*.

> *SimpModeling*: Sketching Implicit Field to Guide Mesh Modeling for 3D Animalmorphic Head Design
>
> [Zhongjin Luo](https://zhongjinluo.github.io/), Jie Zhou, Heming Zhu, Dong Du, [Xiaoguang Han](https://gaplab.cuhk.edu.cn/), [Hongbo Fu](https://sweb.cityu.edu.hk/hongbofu/)

## Introduction

<center>
<img src="./docs/paper-teaser.png" width="100%"/>
</center>
We present *SimpModeling*, a novel sketching system designed for amateur users to create desired animalmorphic heads. It provides two stages for mesh modeling: coarse shape sketching where users may create coarse head models with 3D curve handles (blue), and geometric detail crafting where users may add geometric surface details by drawing sketches (red) on the coarse models. The two animalmorphic head models in this figure were created by a novice user without any 3D modeling experiences in ten minutes. Please refer to our project page for more demonstrations.

##### | [Paper](https://arxiv.org/abs/2108.02548) | [Project](https://zhongjinluo.github.io/SimpModeling/) | 

## Demo

https://github.com/zhongjinluo/SimpModeling/assets/22856460/ac7dd7ba-a533-476f-b9c4-cf157507227e

## Usage

This system has been tested with Python 3.6, PyTorch 1.7.1, CUDA 10.2 on Ubuntu 18.04. 

- Start by cloning this repo:

  ```
  git clone git@github.com:zhongjinluo/SimpModeling.git
  cd SimpModeling
  ```

- Download pre-compiled user interface and checkpoints for backend algorithms from [simpmodeling_files.zip](https://cuhko365-my.sharepoint.com/:u:/g/personal/220019015_link_cuhk_edu_cn/EWSVCrwYdb1Lj3FrQn8a9O0B1xD-2c_IIISJ-1v-ZbtMWQ?e=CHrlxx) and unzip it:

  ```
  unzip app.zip # /path-to-repo/app
  mv simpmodeling_files/coarse/checkpoint_epoch_200.tar /path-to-repo/coarse/experiments/exp_3000v128/checkpoints/
  mv simpmodeling_files/fine/normal/*.pth /path-to-repo/fine/normal/checkpoints/GapMesh/
  mv simpmodeling_files/fine/model/netG_latest /path-to-repo/fine/model/checkpoints/example/
  ```

- After preparing the above file, the directory structure is expected as follows:

  ```
  ├── app
  │   ├── AppRun -> APP_UI_3
  │   ├── APP_UI_3
  │   ├── config.ini
  │   ├── doc
  │   ├── env.sh
  │   ├── lib
  │   ├── pack.sh
  │   ├── plugins
  │   ├── qt.conf
  │   ├── results
  │   ├── run.sh
  │   └── translations
  ├── coarse
  │   ├── data_processing
  │   ├── experiments
  │   │   └── exp_3000v128
  │   │       ├── checkpoints
  │   │       │   └── checkpoint_epoch_200.tar
  │   │       └── val_min=68.npy
  │   ├── models
  │   ├── server.py
  │   └── server.sh
  ├── docs
  ├── fine
  │   ├── model
  │   │   ├── checkpoints
  │   │   │   └── example
  │   │   │       └── netG_latest
  │   │   ├── lib
  │   │   └── PV.json
  │   ├── normal
  │   │   ├── checkpoints
  │   │   │   └── GapMesh
  │   │   │       ├── latest_net_D.pth
  │   │   │       └── latest_net_G.pth
  │   │   ├── models
  │   │   ├── options
  │   │   └── util
  │   ├── server.py
  │   ├── server.sh
  │   ├── sketch2model.py
  │   └── sketch2norm.py
  └── README.md
  ```

- Run the backend servers for two-stage modeling:

  ```
  cd /path-to-repo/coarse && bash server.sh
  cd /path-to-repo/fine && bash server.sh
  ```

- Launch the user interface and enjoy it:

  ```
  cd app/ && bash run.sh
  ```

- If you want to run the backend algorithms on a remote server, you may have to modify  `app/config.ini`. **This repo represents the prototype implementation of our paper. Please use this for research and educational purposes only. This is a research prototype system and made public for demonstration purposes. The user interface runs on Ubuntu 18.04 platforms only and may contain some bugs.**

## Citation

```
@inproceedings{luo2021simpmodeling,
  title={Simpmodeling: Sketching implicit field to guide mesh modeling for 3d animalmorphic head design},
  author={Luo, Zhongjin and Zhou, Jie and Zhu, Heming and Du, Dong and Han, Xiaoguang and Fu, Hongbo},
  booktitle={The 34th Annual ACM Symposium on User Interface Software and Technology},
  pages={854--863},
  year={2021}
}
```


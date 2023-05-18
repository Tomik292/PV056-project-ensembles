# PV056-project

This is a project for the PV056. 

In this project I try to reproduce results from [Domain Adaptive Ensemble Learning (DAEL)](https://ieeexplore.ieee.org/abstract/document/9540778) and compare them with other methods in two problems that the DAEL focuses: Unsupervised Domain adaptation (UDA) and Domain Generalisation (DG)

## Overview
- [UDA](#uda)
- [DG](#dg)
- [Code used](#code-used)
- [Datasets](#datasets)
- [Results](#results)
- [Implementation](#implementation)
  - [Unsupervised Domain Adaptation](#unsupervised-domain-adaptation)
  - [Domain Generalisation](#domain-generaistion)


## UDA

We will compare two methods:

[DAEL](https://ieeexplore.ieee.org/abstract/document/9540778) already mentioned and [M3SDA](https://arxiv.org/abs/1812.01754)

## DG
We will compare two methods:

[DAEL](https://ieeexplore.ieee.org/abstract/document/9540778) already mentioned and [DDAIG](https://arxiv.org/abs/2003.06054)

## Code used:
The paper provides implementations of both UDA and DG in [this gihub repository](https://github.com/KaiyangZhou/Dassl.pytorch). 

How to use this repository and run in on specific datasets is described in the [Implementation](#implementation) part  
(The repository includes also other methods, but we will use just DG and UDA)

## Datasets 
The list below each dataset shows all the different domains

### For UDA:
- Digit-5 - predicts numbers (0-9)
  - MNIST
  - MNIST-M
  - USPS
  - SVHS
  - SYN
- miniDomainNet - A subset of the [DomainNet](http://ai.bu.edu/M3SDA/)
  - Clipart
  - Painting
  - Real
  - Sketch

In the paper the full domain net is also used. To reproduce the training as in the paper, it would take toomuch time and computational resources so I decided to skip it.

### For DG:
- PACS  - dog, elephant, giraffe, guitar, horse, house and person
  - Photo
  - Art painting 
  - Cartoon
  - Sketch
- Office-Home - 65 categories related to office and home objects
  - Artistic 
  - Clipart 
  - Product 
  - Real World

## Results

### Unsupervised domain adaptation **UDA**
Comparison of DAEL and M<sup>3</sup>SDA

**Digit-5** (Accuracy/Macro F1)
| Domain      | DAEL        | M<sup>3</sup>SDA |
| ----------- | ----------- | -----------------|
| MNIST-M     | 88.8%/88.8% | 80.9%/81.4%      |
| MNIST       | 99.4%/99.4% | 99.3%/99.3%      |
| USPS        | 98.8%/98.6% | 98.4%/98.2%      |
| SVHN        | 90.3%/89.4% | 89.5%/88.5%      |
| SYN         | 96.9%/96.9% | 95.9%/95.9%      |

**miniDomainNet** (Accuracy/Macro F1)
| Domain      | DAEL        | M<sup>3</sup>SDA |
| ----------- | ----------- | -----------------|
| Sketch      | 53.0%/51.4% | 47.9%/45.8%      |
| Real        | 66.7%/64.5% | 60.6%/58.9%      |
| Painting    | 54.1%/51.3% | 52.2%/50.9%      |
| Clipart     | 64.9%/63.0% | 60.2%/58.9%      |

### Domain Generalisation **DG**
Comparison of DAEL and DDAIG

**PACS** (Accuracy/Macro F1)
| Domain      | DAEL        | DDAIG            |
| ----------- | ----------- | -----------------|
| Sketch      | 77.3%/77.1% | 73.2%/75.9%      |
| Photo       | 94.9%/94.1% | 94.0%/93.3%      |
| Art Painting| 82.8%/82.2% | 85.4%/85.3%      |
| Cartoon     | 73.8%/74.2% | 72.5%/74.3%      |

**Office-home** (Accuracy/Macro F1)
| Domain      | DAEL        | DDAIG            |
| ----------- | ----------- | -----------------|
| Art         | 59.3%/55.7% | 53.3%/49.6%      |
| Product     | 73.6%/71.4% | 67.9%/66.4%      |
| Real world  | 76.2%/74.6% | 72.6%/70.6%      |
| Clipart     | 55.2%/54.1% | 50.8%/50.3%      |

## Implementation

### Requirements
- python <= 3.8 (I was not able to run CUDA with 3.8 3.9 works fine)
- git
- conda (virtualenv should work too)
- wget

### Preparing repository
```bash
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch
conda create -y -n dassl python=3.8 # 3.9 with cuda
source activate dassl

#For 3.8
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

#For 3.9
pip3 install torch torchvision torchaudio

# Assuming you have the dassl virtual env activated
pip3 install -r requirements.txt
python setup.py develop
```

### Downloading datasets
```bash
pip install gdown
mkdir data
```
**Digit-5**
```bash
mkdir data/digit5
gdown https://drive.google.com/u/0/uc?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm&export=download
unzip Digit-Five.zip -d data/digit5
python digit5.py data/digit5
```
**Mini Domain Net**
```bash
mkdir data/domainnet
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
unzip clipart.zip -d data/domainnet
wget http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip
unzip painting.zip -d data/domainnet
wget http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip
unzip real.zip -d data/domainnet
wget http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
unzip sketch.zip -d data/domainnet
gdown https://drive.google.com/u/0/uc?id=15rrLDCrzyi6ZY-1vJar3u7plgLe4COL7&export=download
unzip splits_mini.zip -d data/domainnet
```
**PACS and Office-Home**

These datasets will be installed when running the training with given parameters

## Unsupervised Domain Adaptation
### Training the DAEL
Create folder to store training info and results in
```bash
mkdir training_results
```
**Running DAEL on Digit-5**
- Source/Target domains to pick from (target domain is always only one):
  - mnist
  - mnist-m
  - usps
  - svhs
  - syn
```bash
python tools/train.py \
--root data \
--trainer DAEL \
--source-domains mnist mnist-m usps svhs\
--target-domains syn \
--dataset-config-file configs/datasets/da/digit5.yaml \
--config-file configs/trainers/da/dael/digit5.yaml \
--output-dir training_results/dael-da-digit5-syn
```
**Running DAEL on miniDomainNet dataset**
- Source/Target domains to pick from (target domain is always only one):
  - clipart
  - painting
  - real
  - sketch
```bash
python tools/train.py \
--root data \
--trainer DAEL \
--source-domains clipart real sketch \
--target-domains painting \
--dataset-config-file configs/datasets/da/mini_domainnet.yaml \
--config-file configs/trainers/da/dael/mini_domainnet.yaml \
--output-dir training_results/dael-da-mini_domainnet-painting
```
### Training the M3SDA
**Running M3SDA on Digit-5 dataset**
- Source/Target domains to pick from (target domain is always only one):
  - mnist
  - mnist-m
  - usps
  - svhs
  - syn
```bash
python tools/train.py \
--root data \
--trainer DAEL \
--source-domains mnist mnist-m usps svhs\
--target-domains syn \
--dataset-config-file configs/datasets/da/digit5.yaml \
--config-file configs/trainers/da/m3sda/digit5.yaml \
--output-dir training_results/m3sda-da-digit5-syn
```
**Running M3SDA on miniDomainNet dataset**
- Source/Target domains to pick from (target domain is always only one):
  - clipart
  - painting
  - real
  - sketch
```bash
python tools/train.py \
--root data \
--trainer M3SDA \
--source-domains clipart real sketch \
--target-domains painting \
--dataset-config-file configs/datasets/da/mini_domainnet.yaml \
--config-file configs/trainers/da/m3sda/mini_domainnet.yaml \
--output-dir training_results/m3sda-da-mini_domainnet-painting
```

## Domain Generaistion
### Training the DAEL
**Running DAEL on PACS dataset**
- Source/Target domains to pick from (target domain is always only one):
  - photo
  - cartoon
  - sketch
  - art_painting
```bash
python tools/train.py \
--root data \
--trainer DAELDG \
--source-domains photo cartoon sketch \
--target-domains art_painting \
--dataset-config-file configs/datasets/dg/pacs.yaml \
--config-file configs/trainers/dg/daeldg/pacs.yaml \
--output-dir training_results/dael-dg-pacs-art_painting
```
**Running DAEL on Office-Home**
- Source/Target domains to pick from (target domain is always only one):
  - real_world
  - clipart
  - product
  - art
```bash
python tools/train.py \
--root data \
--trainer DAELDG \
--source-domains real_world clipart product \
--target-domains art \
--dataset-config-file configs/datasets/dg/office_home_dg.yaml \
--config-file configs/trainers/dg/daeldg/office_home_dg.yaml \
--output-dir training_results/dael-dg-office_home-art
```
### Training the DDAIG 
**Running DDAIG  on PACS dataset**
- Source/Target domains to pick from (target domain is always only one):
  - photo
  - cartoon
  - sketch
  - art_painting
```bash
python tools/train.py \
--root data \
--trainer DDAIG \
--source-domains photo cartoon sketch \
--target-domains art_painting \
--dataset-config-file configs/datasets/dg/pacs.yaml \
--config-file configs/trainers/dg/ddaig/pacs.yaml \
--output-dir training_results/ddaig-dg-pacs-art_painting
```
**Running DDAIG on Office-Home**
- Source/Target domains to pick from (target domain is always only one):
  - real_world
  - clipart
  - product
  - art
```bash
python tools/train.py \
--root data \
--trainer DDAIG \
--source-domains real_world clipart product \
--target-domains art \
--dataset-config-file configs/datasets/dg/office_home_dg.yaml \
--config-file configs/trainers/dg/ddaig/office_home_dg.yaml \
--output-dir training_results/ddaig-dg-office_home-art
```
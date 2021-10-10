<div align="center"><img src="assets/header.png" width="840"></div>

## Introduction

This repo contains the official PyTorch implementation of our ICCV paper
[DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection](https://arxiv.org/abs/2108.09017).

<div align="center"><img src="assets/arch.png" width="800"></div>

## Updates!!
* 【2021/10/10】 We release the official PyTorch implementation of [DeFRCN](https://github.com/er-muyue/DeFRCN).
* 【2021/08/20】 We have uploaded our paper (long version with supplementary material) on [arxiv](https://arxiv.org/abs/2108.09017), review it for more details.

## Quick Start

**1. Check Requirements**
* Linux with Python >= 3.6
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.6 & [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch version.
* CUDA 10.1, 10.2
* GCC >= 4.9

**2. Build DeFRCN**
* Clone Code
  ```angular2html
  git clone https://github.com/er-muyue/DeFRCN.git
  cd DeFRCN
  ```
* Create a virtual environment (optional)
  ```angular2html
  virtualenv defrcn
  cd /path/to/venv/defrcn
  source ./bin/activate
  ```
* Install PyTorch 1.6.0 with CUDA 10.1 
  ```shell
  pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
  ```
* Install Detectron2
  ```angular2html
  python3 -m pip install detectron2==0.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
  ```
  - If you use other version of PyTorch/CUDA, check the latest version of Detectron2 in this page: [Detectron2](https://github.com/facebookresearch/detectron2/releases). 
  - Sorry for that I don’t have enough time to test on more versions, if you run into problems with other versions, please let me know.
* Install other requirements. 
  ```angular2html
  python3 -m pip install -r requirements.txt
  ```

**3. Prepare Data and Weights**
* Data Preparation
  - We evaluate our models on two datasets for both FSOD and G-FSOD settings:

    | Dataset | Size | GoogleDrive | BaiduYun | Note |
    |:---:|:---:|:---:|:---:|:---:|
    |VOC2007| 0.8G |[download](https://drive.google.com/file/d/1BcuJ9j9Mtymp56qGSOfYxlXN4uEVyxFm/view?usp=sharing)|[download](https://pan.baidu.com/s/1kjAmHY5JKDoG0L65T3dK9g)| - |
    |VOC2012| 3.5G |[download](https://drive.google.com/file/d/1NjztPltqm-Z-pG94a6PiPVP4BgD8Sz1H/view?usp=sharing)|[download](https://pan.baidu.com/s/1DUJT85AG_fqP9NRPhnwU2Q)| - |
    |vocsplit| <1M |[download](https://drive.google.com/file/d/1BpDDqJ0p-fQAFN_pthn2gqiK5nWGJ-1a/view?usp=sharing)|[download](https://pan.baidu.com/s/1518_egXZoJNhqH4KRDQvfw)| refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |
    |COCO| ~19G | - | - | download from [offical](https://cocodataset.org/#download)|
    |cocosplit| 174M |[download](https://drive.google.com/file/d/1T_cYLxNqYlbnFNJt8IVvT7ZkWb5c0esj/view?usp=sharing)|[download](https://pan.baidu.com/s/1NELvshrbkpRS8BiuBIr5gA)| refer from [TFA](https://github.com/ucbdrive/few-shot-object-detection#models) |
  - Unzip the downloaded data-source to `datasets` and put it into your project directory:
    ```angular2html
      ...
      datasets
        | -- coco (trainval2014/*.jpg, val2014/*.jpg, annotations/*.json)
        | -- cocosplit
        | -- VOC2007
        | -- VOC2012
        | -- vocsplit
      defrcn
      tools
      ...
    ```
* Weights Preparation
  - We use the imagenet pretrain weights to initialize our model. Download the same models from here: [GoogleDrive](https://drive.google.com/file/d/1rsE20_fSkYeIhFaNU04rBfEDkMENLibj/view?usp=sharing) [BaiduYun](https://pan.baidu.com/s/1IfxFq15LVUI3iIMGFT8slw)
  - The extract code for all BaiduYun link is **0000**

**4. Training and Evaluation**

For ease of training and evaluation over multiple runs, we integrate the whole pipeline of few-shot object detection into one script `run_*.sh`, including base pre-training and novel-finetuning (both FSOD and G-FSOD).
* To reproduce the results on VOC, `EXP_NAME` can be any string (e.g defrcn, or something) and `SPLIT_ID` must be `1 or 2 or 3` (we consider 3 random splits like other papers).
  ```angular2html
  bash run_voc.sh EXP_NAME SPLIT_ID (1, 2 or 3)
  ```
* To reproduce the results on COCO, `EXP_NAME` can be any string (e.g defrcn, or something) 
  ```angular2html
  bash run_coco.sh EXP_NAME
  ```
* Please read the details of few-shot object detection pipeline in `run_*.sh`, you need change `IMAGENET_PRETRAIN*` to your path.

## Results on COCO Benchmark

* Few-shot Object Detection

  |Method| | | mAP<sup>novel</sup> | | | |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | Shot |  1  |  2  |  3  |  5  |  10 |  30 |
  |[FRCN-ft](https://arxiv.org/abs/1506.01497)|1.0*|1.8*|2.8*|4.0*|6.5|11.1|
  |[FSRW](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kang_Few-Shot_Object_Detection_via_Feature_Reweighting_ICCV_2019_paper.pdf)|-|-|-|-|5.6|9.1|
  |[MetaDet](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Meta-Learning_to_Detect_Rare_Objects_ICCV_2019_paper.pdf)|-|-|-|-|7.1|11.3|
  |[MetaR-CNN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Meta_R-CNN_Towards_General_Solver_for_Instance-Level_Low-Shot_Learning_ICCV_2019_paper.pdf)|-|-|-|-|8.7|12.4|
  |[TFA](http://proceedings.mlr.press/v119/wang20j/wang20j.pdf)|4.4*|5.4*|6.0*|7.7*|10.0|13.7|
  |[MPSR](https://arxiv.org/abs/2007.09384)|5.1*|6.7*|7.4*|8.7*|9.8|14.1|
  |[FSDetView](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620188.pdf)|4.5|6.6|7.2|10.7|12.5|14.7|
  |[DeFRCN](https://arxiv.org/abs/2108.09017) (Our Paper)|**9.3**|**12.9**|**14.8**|**16.1**|**18.5**|**22.6**|
  |[DeFRCN](https://arxiv.org/abs/2108.09017) (This Repo) |**9.7**|**13.1**|**14.5**|**15.6**|**18.4**|**22.6**|
  
* Generalized Few-shot Object Detection

  |Method| | | mAP<sup>novel</sup>| | | |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | Shot |  1  |  2  |  3  |  5  |  10 |  30 |
  |[FRCN-ft](https://arxiv.org/abs/1506.01497)|1.7|3.1|3.7|4.6|5.5|7.4|
  |[TFA](http://proceedings.mlr.press/v119/wang20j/wang20j.pdf)|1.9|3.9|5.1|7|9.1|12.1|
  |[FSDetView](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620188.pdf)|3.2|4.9|6.7|8.1|10.7|15.9|
  |[DeFRCN](https://arxiv.org/abs/2108.09017) (Our Paper)|**4.8**|**8.5**|**10.7**|**13.6**|**16.8**|**21.2**|
  |[DeFRCN](https://arxiv.org/abs/2108.09017) (This Repo) |**4.8**|**8.5**|**10.7**|**13.5**|**16.7**|**21.0**|
  
- \* indicates that the results are reproduced by us with their source code.
- It's normal to observe -0.3~+0.3AP noise between your results and this repo. 
- The results of mAP<sup>base</sup> and mAP<sup>all</sup> for G-FSOD are list here [GoogleDrive](https://drive.google.com/file/d/1WUM2X-pPzox2fQz4aLi3YzxGgscpnoHU/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1XsSa7vxKDWZzSbPU-s0aiA).  
- If you have any problem of above results in this repo, you can download *configs and train logs* from [GoogleDrive](https://drive.google.com/file/d/1WUM2X-pPzox2fQz4aLi3YzxGgscpnoHU/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1XsSa7vxKDWZzSbPU-s0aiA).

## Results on VOC Benchmark

* Few-shot Object Detection
 
  |Method| | |Split-1| | | | |Split-2| | | | |Split-3| | |
  |:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
  |Shot|1|2|3|5|10|1|2|3|5|10|1|2|3|5|10|
  |[YOLO-ft](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kang_Few-Shot_Object_Detection_via_Feature_Reweighting_ICCV_2019_paper.pdf)|6.6|10.7|12.5|24.8|38.6|12.5|4.2|11.6|16.1|33.9|13.0|15.9|15.0|32.2|38.4|
  |[FRCN-ft](https://arxiv.org/abs/1506.01497)|13.8|19.6|32.8|41.5|45.6|7.9|15.3|26.2|31.6|39.1|9.8|11.3|19.1|35.0|45.1|
  |[FSRW](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kang_Few-Shot_Object_Detection_via_Feature_Reweighting_ICCV_2019_paper.pdf)|14.8|15.5|26.7|33.9|47.2|15.7|15.2|22.7|30.1|40.5|21.3|25.6|28.4|42.8|45.9|
  |[MetaDet](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Meta-Learning_to_Detect_Rare_Objects_ICCV_2019_paper.pdf)|18.9|20.6|30.2|36.8|49.6|21.8|23.1|27.8|31.7|43.0|20.6|23.9|29.4|43.9|44.1|
  |[MetaR-CNN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Meta_R-CNN_Towards_General_Solver_for_Instance-Level_Low-Shot_Learning_ICCV_2019_paper.pdf)|19.9|25.5|35.0|45.7|51.5|10.4|19.4|29.6|34.8|45.4|14.3|18.2|27.5|41.2|48.1|
  |[TFA](http://proceedings.mlr.press/v119/wang20j/wang20j.pdf)|39.8|36.1|44.7|55.7|56.0|23.5|26.9|34.1|35.1|39.1|30.8|34.8|42.8|49.5|49.8|
  |[MPSR](https://arxiv.org/abs/2007.09384)|41.7|-|51.4|55.2|61.8|24.4|-|39.2|39.9|47.8|35.6|-|42.3|48.0|49.7|
  |[DeFRCN](https://arxiv.org/abs/2108.09017) (Our Paper)|**53.6**|**57.5**|**61.5**|**64.1**|**60.8**|**30.1**|**38.1**|**47.0**|**53.3**|**47.9**|**48.4**|**50.9**|**52.3**|**54.9**|**57.4**|
  |[DeFRCN](https://arxiv.org/abs/2108.09017) (This Repo)|**55.1**|**57.4**|**61.1**|**64.6**|**61.5**|**32.1**|**40.5**|**47.9**|**52.9**|**47.5**|**48.9**|**51.9**|**52.3**|**55.7**|**59.0**|

* Generalized Few-shot Object Detection

  |Method| | |Split-1| | | | |Split-2| | | | |Split-3| | |
  |:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
  |Shot|1|2|3|5|10|1|2|3|5|10|1|2|3|5|10|
  |[FRCN-ft](https://arxiv.org/abs/1506.01497)|9.9|15.6|21.6|28.0|52.0|9.4|13.8|17.4|21.9|39.7|8.1|13.9|19|23.9|44.6|
  |[FSRW](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kang_Few-Shot_Object_Detection_via_Feature_Reweighting_ICCV_2019_paper.pdf)|14.2|23.6|29.8|36.5|35.6|12.3|19.6|25.1|31.4|29.8|12.5|21.3|26.8|33.8|31.0|
  |[TFA](http://proceedings.mlr.press/v119/wang20j/wang20j.pdf)|25.3|36.4|42.1|47.9|52.8|18.3|27.5|30.9|34.1|39.5|17.9|27.2|34.3|40.8|45.6|
  |[FSDetView](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620188.pdf)|24.2|35.3|42.2|49.1|57.4|21.6|24.6|31.9|37.0|45.7|21.2|30.0|37.2|43.8|49.6|
  |[DeFRCN](https://arxiv.org/abs/2108.09017) (Our Paper)|**40.2**|**53.6**|**58.2**|**63.6**|**66.5**|**29.5**|**39.7**|**43.4**|**48.1**|**52.8**|**35.0**|**38.3**|**52.9**|**57.7**|**60.8**|
  |[DeFRCN](https://arxiv.org/abs/2108.09017) (This Repo)|**43.8**|**57.5**|**61.4**|**65.3**|**67.0**|**31.5**|**40.9**|**45.6**|**50.1**|**52.9**|**38.2**|**50.9**|**54.1**|**59.2**|**61.9**|

* Note that we change the λ<sup>GDL-RCNN</sup> for VOC to 0.001 (0.01 in paper) and get better performance, check the configs for more details.
* The results of mAP<sup>base</sup> and mAP<sup>all</sup> for G-FSOD are list here [GoogleDrive](https://drive.google.com/file/d/1Ff5jP4PCDDPQ7lzsageZsauFWer73QIl/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1FQq_2EgqlmzFv8cANOFYmA).
* If you have any problem of above results in this repo, you can download *configs and logs* from [GoogleDrive](https://drive.google.com/file/d/1Ff5jP4PCDDPQ7lzsageZsauFWer73QIl/view?usp=sharing), [BaiduYun](https://pan.baidu.com/s/1FQq_2EgqlmzFv8cANOFYmA).

## Acknowledgement
This repo is developed based on [TFA](https://github.com/Megvii-BaseDetection/cvpods) and [Detectron2](https://github.com/facebookresearch/detectron2). Please check them for more details and features.

## Citing
If you use this work in your research or wish to refer to the baseline results published here, please use the following BibTeX entries:
```
@inproceedings{qiao2021defrcn,
  title={DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection},
  author={Qiao, Limeng and Zhao, Yuxuan and Li, Zhiyuan and Qiu, Xi and Wu, Jianan and Zhang, Chi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={8681--8690},
  year={2021}
}
```

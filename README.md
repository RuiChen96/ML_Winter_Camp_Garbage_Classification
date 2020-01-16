# ML_Winter_Camp_2020_Garbage_Classification

##### Team: DuckDuckGo

##### Members: Rui, Jing and Hanyu

---

#### 0. Environment Setup

```bash
conda create -n garbage_cls python=3.6
conda activate garbage_cls  # or source activate garbage_cls
pip install -r requirements.txt
```

#### 1. Data Preparation

```bash
cd training_scripts/
mkdir data
mkdir model_save
cd data/
```

Then, Download [the garbage dataset](https://modelarts-competitions.obs.cn-north-1.myhuaweicloud.com/garbage_classify/dataset/garbage_classify_v2.zip)

```bash
wget https://modelarts-competitions.obs.cn-north-1.myhuaweicloud.com/garbage_classify/dataset/garbage_classify_v2.zip
unzip garbage_classify_v2.zip
rm garbage_classify_v2.zip
```

Finally, run the pre-processing script

```bash
cd ..
python pre_processing.py
```

#### 2. Model Training

```bash
python train.py
```

We also provide [the pre-trained model]() on the garbage dataset, which has `top-1` accuracy of `93.66%` and `top-5` accuracy of `99.89%`.

#### 3. WeChat Deployment

```bash
cd ..
python app.py
```

Our deployment actually borrow the help of [frp (fast reverse proxy) (in Chinese: 内网穿透)](https://github.com/fatedier/frp). 
If both of your `port 80` and `port 443` are blocked owning to firewall issue, you may check this out for your own WeChat deployment.

#### Acknowledgement

We borrow tons of code from [the second place solution](https://github.com/ikkyu-wen/huawei-garbage) of garbage classification competition held by Huawei.

#### Reference

[1] Training code base: [huawei-garbage](https://github.com/ikkyu-wen/huawei-garbage)

[2] Fast reverse proxy: [frp](https://github.com/fatedier/frp)

[3] Mahajan, Dhruv, et al. "[Exploring the limits of weakly supervised pretraining](https://arxiv.org/abs/1805.00932)." ECCV. 2018.

[4] He, Kaiming, et al. "[Deep residual learning for image recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)." CVPR. 2016.

[5] Krizhevsky, Alex, et al. "[Imagenet classification with deep convolutional neural networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)." NeurIPS. 2012.

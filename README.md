
# 简介 Introduction
本仓库为一个开发模板，用以简便开发
需将lib目录标记为 源 根
# 快速开始 Quick start

## 安装 Installation
1. 克隆此存储库，我们将您克隆的 Neural Network Development Template 目录命名为 {PROJECT_ROOT} 
2. 安装依赖项。
3. 下载预训练模型。请将它们下载到{PROJECT_ROOT}模型下，并使它们看起来像这样：
   ```
   ${PROJECT_ROOT}/models
   └── pytorch
       └── resnet
           ├── resnet152-b121ed2d.pth
           └── resnet50-19c8e357.pth

   ```
   可以从以下链接下载它们：
   https://onedrive.live.com/?authkey=%21AF9rKCBVlJ3Qzo8&id=93774C670BD4F835%21930&cid=93774C670BD4F835
   
   

4. 初始化 output (训练模型输出目录)和 log (tensorboard日志目录)目录。

   你的目录树应该像这样
   ```
   ${PROJECT_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── weights
   ├── output
   ├── pose_estimation
   ├── README.md
   ├── requirements.txt
   ```

## 数据准备 Data preparation
下载数据集，并将标签转换为json格式，
以 MPII 数据集为例，将它们提取到 {PROJECT_ROOT}/data 下，并使它们看起来像这样：
```
${PROJECT_ROOT}
├── data
├── ├── MPII
    ├── ├── labels
        |   ├── gt_valid.mat
        |   ├── test.json
        |   ├── train.json
        |   ├── validate.json
        |   
        ├── images
            ├── 000001163.jpg
            ├── 000003072.jpg
```

如果将图像文件压缩为单个 zip 文件，则应按如下方式组织数据：

```
${PROJECT_ROOT}
├── data
└── ├── MPII
    └── ├── annot
        |   ├── gt_valid.mat
        |   ├── test.json
        |   ├── train.json
        |   └── valid.json
        |   
        └── images.zip
            └── images
                ├── 000001163.jpg
                ├── 000003072.jpg
```

对于一些先验数据，可以将其放入 ${PROJECT_ROOT}/data/prior/ 下

## 训练和测试 Training and Testing
数据集上的训练和测试
```
python run/train.py --cfg experiments/resnet50/256_fusion.yaml
python run/validate.py --cfg experiments/resnet50/256_fusion.yaml
python run/test.py --cfg experiments/resnet50/256_fusion.yaml
```



# ------------------------------------------------------------------------------
# Written by Miao
# ------------------------------------------------------------------------------

import json
import os

import cv2
import copy
import os.path as osp

from torch.utils.data import Dataset


class ExampleDataset(Dataset):

    def __init__(self, cfg, source, subset, transform=None):
        """
        Args:
            cfg: 配置文件
            source: 数据集名称，如 MPII、h36m 等
            subset: 数据集子集名称，如 train、test 等
            transform: 变形器，用于对图片进行仿射变换
        """

        self.source = source
        self.subset = subset

        self.root = cfg.DATASET.ROOT
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.image_size = cfg.NETWORK.IMAGE_SIZE

        self.transform = transform
        self.labels = self._get_labels()

    def _get_labels(self):
        label_path = os.path.join(self.root, self.source, 'labels',
                                  self.subset + '.json')

        with open(label_path) as label_file:
            labels = json.load(label_file)

        return labels


    def __len__(self, ):
        return len(self.labels)

    def __getitem__(self, idx):
        label = copy.deepcopy(self.labels[idx])

        image_dir = 'images.zip@' if self.data_format == 'zip' else ''
        image_path = osp.join(
            self.root, label['source'], image_dir, 'images', label['image_name'])

        if self.data_format == 'zip':
            from utils import zipreader
            image = zipreader.imread(
                image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            image = cv2.imread(
                image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # TODO  将图片转为输入尺寸，或进行仿射变换以增强数据集
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # TODO  将 label 处理为所需格式
        target = [
            label['center'].copy(),
            label['height'].copy(),
            label['width'].copy()
        ]

        confidence = label['Confidence'].copy(),

        return image, target, confidence

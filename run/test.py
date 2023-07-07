# ------------------------------------------------------------------------------
# Written by Miao
# ------------------------------------------------------------------------------

import torch
import argparse

from _init_paths import add_path
from core.config import config
from core.config import update_config
from utils.utils import create_logger

from dataset import ExampleDataset
from models import ExampleModel


def parse_args():
    parser = argparse.ArgumentParser(description='Test network')
    parser.add_argument(
        '--cfg', help='configuration file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    add_path()
    args = parse_args()
    device = torch.device('cuda:0')
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'test')

    # TODO  选择测试集和模型
    test_dataset = ExampleDataset(config, )
    model = ExampleModel(config, )
    model.init_weights(config.NETWORK.PRETRAINED)

    # TODO  测试代码
    for item in test_dataset:
        prediction = model(item, config)
        print(prediction)



if __name__ == '__main__':
    main()

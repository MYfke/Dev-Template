# ------------------------------------------------------------------------------
# Written by Miao
# ------------------------------------------------------------------------------

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from _init_paths import add_path
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import MSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import create_logger

from dataset import ExampleDataset
from models import ExampleModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument('--frequent', help='frequency of logging', default=config.PRINT_FREQ, type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)

    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--data-format', help='data format', type=str, default='')

    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)
    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.workers:
        config.WORKERS = args.workers


def main():
    add_path()
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # TODO 选择训练模型
    model = ExampleModel(config, )

    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', config.MODEL + '.py'),
        final_output_dir)
    shutil.copy2(args.cfg, final_output_dir)
    logger.info(pprint.pformat(model))
    print(model)

    # TODO  TensorBoardX代码
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # TODO  选择损失函数、优化器、学习率
    criterion = MSELoss(use_target_confidence=config.LOSS.USE_CONFIDENCE).cuda()
    optimizer = get_optimizer(config, model)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    # TODO  恢复已终止的训练代码
    start_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.RESUME:
        start_epoch, model, optimizer = load_checkpoint(model, optimizer, final_output_dir)

    # TODO  数据加载代码
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = ExampleDataset(
        config,
        config.DATASET.SOURCE,
        config.DATASET.TRAIN_SUBSET,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    validate_dataset = ExampleDataset(
        config,
        config.DATASET.SOURCE,
        config.DATASET.VALIDATE_SUBSET,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(
        validate_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    best_perf, best_model = 0.0, False
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # TODO  训练代码和验证代码
        train(config, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, writer_dict)

        perf_indicator = validate(config, validate_loader, model, criterion,
                                  final_output_dir, writer_dict)

        if perf_indicator > best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': get_model_name(config),
            'state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('saving final model state to {}'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)


if __name__ == '__main__':
    main()

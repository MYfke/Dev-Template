# ------------------------------------------------------------------------------
# Written by Miao
# ------------------------------------------------------------------------------

import torch
import time
import logging
import math
import numpy as np

from utils.transforms import transform_preds


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """
    计算并存储平均值和当前值
    Computes and stores the average and current value
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_max_preds(preds):
    """
    TODO  非极大抑制代码
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """

    return preds



def evaluate(output, target):
    """
    TODO  计算准确度与其他性能指标
    """

    acc = target - output
    perf_indicator = target - output

    return acc, perf_indicator


def train(config, loader, model, criterion, optim, epoch, final_output_dir, writer_dict):
    batch_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()
    data_time = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, confidence) in enumerate(loader):
        data_time.update(time.time() - end)
        output = model(input)

        loss = 0
        target_cuda = []
        for o, t, w in zip(output, target, confidence):
            t = t.cuda(non_blocking=True)
            w = w.cuda(non_blocking=True)
            target_cuda.append(t)
            loss += criterion(o, t, w)
        target = target_cuda

        # 反向传播
        optim.zero_grad()
        loss.backward()
        optim.step()

        acc, perf_indicator = evaluate(output, target)

        # TODO  将 训练 结果可视化和输出日志文件
        nimgs = input.size(0)
        avg_loss.update(loss.item(), nimgs)
        avg_acc.update(acc, nimgs)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t' \
                  'Memory {memory:.1f}'.format(
                    epoch, i, len(loader),
                    batch_time=batch_time,
                    speed=batch_time.val / len(input),
                    data_time=data_time,
                    loss=avg_acc,
                    acc=avg_acc,
                    memory=gpu_memory_usage)
            logger.info(msg)

            # TODO  更新TensorBoardX数据
            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', avg_loss.val, global_steps)
            writer.add_scalar('train_acc', avg_acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1


def validate(config, loader, model, criterion, final_output_dir, writer_dict):
    batch_time = AverageMeter()
    avg_loss = AverageMeter()
    avg_acc = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target, confidence) in enumerate(loader):
            output = model(input)

            loss = 0
            target_cuda = []
            for o, t, w in zip(output, target, confidence):
                t = t.cuda(non_blocking=True)
                w = w.cuda(non_blocking=True)
                target_cuda.append(t)
                loss += criterion(o, t, w)
            target = target_cuda

            acc, perf_indicator = evaluate(output, target)

            # TODO  将 验证 结果可视化和输出日志文件
            nimgs = input.size(0)
            avg_loss.update(loss.item(), nimgs)
            avg_acc.update(acc, nimgs)
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        i, len(loader),
                        batch_time=batch_time,
                        loss=avg_acc,
                        acc=avg_acc)
                logger.info(msg)

    return perf_indicator

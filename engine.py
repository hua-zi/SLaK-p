# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
# import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils

import paddle

def train_one_epoch(model: paddle.nn.Layer, criterion: paddle.nn.Layer,
                    data_loader: Iterable, optimizer: paddle.optimizer.Optimizer,
                    device: paddle.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, mask=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('loss', utils.SmoothedValue(window_size=4, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # print('mmmmmmmmmmmmmmm')
    # for i in metric_logger.meters:
    #     print(i)

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200
    
    optimizer.clear_grad()

    # for batch in metric_logger.log_every(data_loader, 10, header):
    #     # print('batch',batch[0].shape,batch[1],batch[0][:,0,0,:5])
    #     images = batch[0]
    #     targets = batch[-1]
    #     images, targets = mixup_fn(images, targets)
    #     output = model(images)
    #     # print('--------------',output.shape, targets.shape)
    #     loss = criterion(output, targets)
    #     loss /= update_freq
    #     # loss.backward()
    #     # optimizer.step()
    #     # print(loss)
    #     metric_logger.update(loss=loss.item())
    #     min_lr = 10.
    #     max_lr = 0.
    #     for group in optimizer._param_groups:
    #         # min_lr = min(min_lr, group["lr"])
    #         # max_lr = max(max_lr, group["lr"])
    #         lr = optimizer._learning_rate
    #         min_lr = min(min_lr, lr)
    #         max_lr = max(max_lr, lr)
    #     metric_logger.update(lr=max_lr)
    #     metric_logger.update(min_lr=min_lr)
    #     # return 1
    #     # acc1, acc5 = accuracy(output, targets, topk=(1, 5))
    #     # print('output',output.shape,output[:,:5])
    #     # print(f'loss:{loss.item():f}  acc1:{acc1.item():f}  acc5:{acc5.item():f}')

    # # gather the stats from all processes
    # # metric_logger.synchronize_between_processes()
    # print('* loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))
    # return 1
    
    flag = 0
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        flag += 1;
        if flag > 10:
            break
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer._param_groups):
                if lr_schedule_values is not None:
                    param_group["learning_rate"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        # samples = samples.to(device, non_blocking=True)
        # targets = targets.to(device, non_blocking=True)
        samples = paddle.to_tensor(samples,place=device)
        targets = paddle.to_tensor(targets,place=device)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with paddle.amp.auto_cast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output = model(samples)
            loss = criterion(output, targets)

        # print('losslllllllllllll', loss.shape, loss)
        loss_value = loss.item()
        
        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.clear_grad()
                if model_ema is not None:
                    model_ema.update(model, mask)

        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                if mask:
                    mask.step()
                else:
                    optimizer.step()
                optimizer.clear_grad()
                if model_ema is not None:
                    model_ema.update(model, mask)
        
        paddle.device.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        
        for group in optimizer._param_groups:
            # min_lr = min(min_lr, group["lr"])
            # max_lr = max(max_lr, group["lr"])
            lr = optimizer._learning_rate
            min_lr = min(min_lr, lr)
            max_lr = max(max_lr, lr)
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer._param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        # return 1
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
        # return 1
        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            
    # return 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@paddle.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = paddle.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    # switch to evaluation mode
    model.eval()

    # print('--------hhhhhhhh------data_loader',type(data_loader), len(data_loader))
    flag = 0
    for batch in metric_logger.log_every(data_loader, 10, header):
        flag += 1
        if flag>10:
            break
        images = batch[0]
        target = batch[-1]

        # images = images.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)
        images = paddle.to_tensor(images,place=device)
        target = paddle.to_tensor(target,place=device)

        # compute output
        if use_amp:
            # with torch.cuda.amp.autocast():
            with paddle.amp.auto_cast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # print("criterion = %s" % str(criterion))
        # print('--------------',use_amp,output.shape, target.shape)
        # print('output',output.shape,output[:,:5])
        # print(f'loss:{loss.item():.3f}  acc1:{acc1.item():.3f}  acc5:{acc5.item():.3f}')
        # return 1

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

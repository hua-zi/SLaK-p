import argparse
import datetime
import numpy as np
import time
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
# from model_sema import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner
from datasets import build_dataset, DistributedSampler, DataLoader
from engine import train_one_epoch, evaluate
from utils import NativeScalerWithGradNormCount as NativeScaler
# from sparse_core import Masking, CosineDecay

import utils
import models.SLaK

import paddle

# 导入reprod_log中的ReprodLogger类
# from reprod_log import ReprodLogger

# reprod_logger = ReprodLogger()
# 组网并初始化
# model = alexnet(pretrained="../../weights/alexnet_paddle.pdparams" num_classes=1000)
model = create_model(
        'SLaK_tiny',
        pretrained=False,
        num_classes=1000,
        drop_path_rate=0.2,
        layer_scale_init_value=1e-06,
        head_init_scale=1.0,
        kernel_size=[51, 49, 47, 13, 5],
        width_factor=1.3,
        Decom=True,
        bn = True
    )
checkpoint = paddle.load('./path/to/checkpoint/SLaK_tiny_checkpoint.pdparams')
model.set_state_dict(checkpoint['model'])
model.eval()

# 读入fake data并转换为tensor，这里也可以固定seed在线生成fake data
# fake_data = np.load("../../fake_data/fake_data.npy")
np.random.seed(10)
fake_data = np.random.randint(low=0,high=255,size=(1,3,244,244),dtype='int32')
fake_data = paddle.to_tensor(fake_data, dtype='float32')
target = paddle.to_tensor([0],dtype='int64')
# print(fake_data)
# 模型前向
out = model(fake_data)
# 计算 loss
criterion = paddle.nn.CrossEntropyLoss()
loss = criterion(out, target)
print('--------------')
print(out, loss)
# 保存前向结果，对于不同的任务，需要开发者添加。
# reprod_logger.add("logits", out.cpu().detach().numpy())
# reprod_logger.add("loss", loss.cpu().detach().numpy())
# reprod_logger.save("forward_paddle.npy")
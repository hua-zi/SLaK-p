import paddle
from paddle.vision import datasets, transforms
import paddle.vision.transforms as T
from datasets import build_dataset, DistributedSampler, DataLoader

import numpy as np

from timm.models import create_model
import models.SLaK

if __name__=='__main__':

    # dataset_val = paddle.vision.datasets.ImageFolder('/home/aistudio/work/SLaK/path/to/imagenet/val', transform=transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),]))

    # data_loader_val = paddle.io.DataLoader(dataset_val,batch_size=64)
    # # data_loader_val = DataLoader(
    # #         dataset_val, 
    # #         batch_size=64,
    # #         # # sampler=sampler_val,
    # #         # batch_size=int(1.5 * args.batch_size),
    # #         # num_workers=args.num_workers,
    # #         # pin_memory=args.pin_mem,
    # #         drop_last=False
    # #     )
    # # print(type(dataset_val),type(dataset_val[0]),len(dataset_val[0]),type(dataset_val[0][0]),dataset_val[0][0].dtype,dataset_val[0][0].shape)
    # # print(len(data_loader_val))
    # for data in data_loader_val:
    #     # print(type(data[0]),len(data[0]),data[0].dtype)
    #     print('hhhh')
    #     break

    # checkpoint = paddle.load('./path/to/checkpoint/SLaK_tiny_checkpoint.pdparams')
    checkpoint = paddle.load('./mlp.pdparams')
    model = create_model(
        'SLaK_tiny',
        pretrained=False,
        num_classes=1000,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        kernel_size=[51,49,47,13,5],
        width_factor=1.0,
        Decom=False,
        bn = True
        )
    # model_without_ddp = model

    for key in checkpoint:
        print(key, '|', checkpoint[key].shape)

    # model.set_state_dict(checkpoint['model'])
    model.set_state_dict(checkpoint)
    # print(model)
    model.eval()
    x = paddle.ones([1,3,512,512])
    output = model(x)
    print(output[0][:5])

    # paddle.save(model.state_dict(), 'mlp.pdparams')

    print('end')

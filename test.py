import os
import csv
import pdb
import time
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import torchvision
from tqdm import tqdm, trange


from models.spatial_transforms import *
from models.temporal_transforms import *
from data import dataset_jester, dataset_EgoGesture, dataset_sthv2
import utils as utils
from models import models as TSN_model
import argparse

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings("ignore")



def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=str, default='2')

    # args for dataloader
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--clip_len', type=int, default=8)
    
    # args for preprocessing
    parser.add_argument('--shift_div', default=8, type=int)
    parser.add_argument('--is_shift', action="store_true")
    parser.add_argument('--base_model', default='resnet50', type=str)
    parser.add_argument('--dataset', default='EgoGesture', type=str)

    # args for testing 
    parser.add_argument('--test_crops', default=1, type=int)
    parser.add_argument('--scale_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--clip_num', type=int, default=10)

    args = parser.parse_args()
    return args

args = parse_opts()

params = dict()
if args.dataset == 'EgoGesture':
    params['num_classes'] = 83
elif args.dataset == 'jester':
    params['num_classes'] = 27
elif args.dataset == 'sthv2':
    params['num_classes'] = 174



annot_path = 'data/{}_annotation'.format(args.dataset)
label_path = '/home/raid/zhengwei/{}/'.format(args.dataset) # for submitting testing results
# annot_path = '/home/raid/zhengwei/kinetic-700'

os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
device = 'cuda:0'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def inference(model, val_dataloader):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(val_dataloader)):
            data_time.update(time.time() - end)
            if args.dataset == 'EgoGesture':
                rgb, depth, labels = inputs[0], inputs[1], inputs[2]
                rgb = rgb.to(device, non_blocking=True).float()
                depth = depth.to(device, non_blocking=True).float()
                nb, n_clip, nt, nc, h, w = rgb.size()
                rgb = rgb.view(-1, nt//args.test_crops, nc, h, w) # n_clip * nb (1) * crops, T, C, H, W
                outputs = model(rgb)
                outputs = outputs.view(nb, n_clip*args.test_crops, -1)
                outputs = F.softmax(outputs, 2)
            else:
                # pdb.set_trace()
                rgb, labels = inputs[0], inputs[1]
                rgb = rgb.to(device, non_blocking=True).float()
                nb, n_clip, nt, nc, h, w = rgb.size()
                rgb = rgb.view(-1, nt//args.test_crops, nc, h, w)
                outputs = model(rgb)
                outputs = outputs.view(nb, n_clip*args.test_crops, -1)
                outputs = F.softmax(outputs, 2)
            labels = labels.to(device, non_blocking=True).long()


            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data.mean(1), labels, topk=(1, 5))
            top1.update(prec1.item(), labels.size(0))
            top5.update(prec5.item(), labels.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (step+1) % 100 == 0:
                print_string = ('Top-1: {top1_acc.avg:.2f}, ' 
                                'Top-5: {top5_acc.avg:.2f}'
                                .format(
                                        top1_acc=top1,
                                        top5_acc=top5)
                                )
                print(print_string) 

    print_string = ('Top-1: {top1_acc:.2f}, ' 'Top-5: {top5_acc:.2f}'.format(
        top1_acc=top1.avg,
        top5_acc=top5.avg)
        )
    print(print_string)





if __name__ == '__main__':
    if args.dataset == 'EgoGesture':
        cropping = torchvision.transforms.Compose([
            GroupScale([224, 224])
        ])
    else:
        if args.test_crops == 1:
            cropping = torchvision.transforms.Compose([
                GroupScale(args.scale_size),
                GroupCenterCrop(args.crop_size)
            ])
        elif args.test_crops == 3:
            cropping = torchvision.transforms.Compose([
                GroupFullResSample(args.crop_size, args.scale_size, flip=False)
            ])
        elif args.test_crops == 5: 
            cropping = torchvision.transforms.Compose([
                GroupOverSample(args.crop_size, args.scale_size, flip=False)
            ])


    input_mean=[.485, .456, .406]
    input_std=[.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)


    # for mulitple clip test, use random sampling;
    # for single clip test, use middle sampling  
    spatial_transform  = torchvision.transforms.Compose([
                            cropping,
                            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                            normalize
                            ])
    temporal_transform = torchvision.transforms.Compose([
            TemporalUniformCrop_train(args.clip_len)
        ])    


    checkpoint_path = '{}-{}/2021-03-11-21-49-43/clip_len_{}frame_sample_rate_1_checkpoint.pth.tar'.format(args.dataset, args.base_model,
                                                                                                                 args.clip_len)
    cudnn.benchmark = True
    model = TSN_model.TSN(params['num_classes'], args.clip_len, 'RGB', 
                        is_shift = args.is_shift,
                        base_model=args.base_model, 
                        shift_div = args.shift_div, 
                        img_feature_dim = args.crop_size,
                        consensus_type='avg',
                        fc_lr5 = True)

    pretrained_dict = torch.load(checkpoint_path, map_location='cpu')
    print("load checkpoint {}".format(checkpoint_path))
    model.load_state_dict(pretrained_dict['state_dict'])
    # model = nn.DataParallel(model)  # multi-Gpu
    model = model.to(device)


    if args.dataset == 'jester':
        val_dataloader = DataLoader(dataset_jester.dataset_video_inference(annot_path, 'val', clip_num=args.clip_num, 
                                                                            spatial_transform=spatial_transform, 
                                                                            temporal_transform = temporal_transform),
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers)
    elif args.dataset == 'sthv2':
        val_dataloader = DataLoader(dataset_sthv2.dataset_video_inference(annot_path, 'val', clip_num=args.clip_num, 
                                                                                spatial_transform=spatial_transform, 
                                                                                temporal_transform = temporal_transform),
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers)

    elif args.dataset == 'EgoGesture':
        val_dataloader = DataLoader(dataset_EgoGesture.dataset_video_inference(annot_path, 'test', clip_num=args.clip_num, 
                                                                                spatial_transform=spatial_transform, 
                                                                                temporal_transform = temporal_transform),
                                    batch_size=args.batch_size, 
                                    num_workers=args.num_workers)

    inference(model, val_dataloader)   


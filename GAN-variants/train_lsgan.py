"""
A pytorch implementation of Least-square GAN from Xudong Mao et al. https://arxiv.org/pdf/1611.04076v2.pdf

some code borrowed from pytorch/examples/dcgan/main.py

Ran Xu
3/5/2017
"""

from __future__ import print_function
import pdb
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='number of G filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='number of D filters in first conv layer')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

opt.manualSeed = random.randint(1, 10000)
print('Random Seed: ', opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print('Warning: You have a cuda device, so you should probably run with --cuda')

if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=False,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
else:
    raise ValueError('dataset not available, only have cifar10 for now')

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


num_gpu = int(opt.ngpu)
num_z = int(opt.nz)
num_gf = int(opt.ngf)
num_df = int(opt.ndf)
num_c = 3


# weight initialization for Gnet and Dnet
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):

    def __init__(self, num_gpu):
        super(_netG, self).__init__()
        self.num_gpu = num_gpu
        self.main = nn.Sequential(
            # input
            nn.ConvTranspose2d(num_z, num_gf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_gf * 8),
            nn.ReLU(True),
            # layer 1
            nn.ConvTranspose2d(num_gf * 8, num_gf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gf * 4),
            nn.ReLU(True),
            # layer 2
            nn.ConvTranspose2d(num_gf * 4, num_gf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gf * 2),
            nn.ReLU(True),
            # layer 3
            nn.ConvTranspose2d(num_gf * 2, num_gf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_gf),
            nn.ReLU(True),
            # layer 4
            nn.ConvTranspose2d(num_gf, num_c, 4, 2, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            gpu_ids = range(self.num_gpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)


netG = _netG(num_gpu)
pdb.set_trace()
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

print(netG)

pdb.set_trace()

# python train_lsgan.py --dataset cifar10 --dataroot /Users/ranxu/Documents/dataset/cifar10








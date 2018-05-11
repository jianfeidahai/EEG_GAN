import utils, torch, time, os, pickle, imageio, math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from torch.autograd import Variable, grad
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from utils import Flatten


class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.input_dim = 3
		self.input_height = 64
		self.input_width = 64
		self.output_dim = 300

		self.conv = nn.Sequential(
			nn.Conv2d(self.input_dim, 64, 3, 4, 2, bias=True),
			nn.InstanceNorm2d(64, affine=True),
			nn.ReLU(),

			nn.Conv2d(64, 128, 4, 2, 1, bias=True),
			nn.InstanceNorm2d(128, affine=True),
			nn.ReLU(),

			nn.Conv2d(128,256, 4, 2, 1, bias=True),
			nn.InstanceNorm2d(256, affine=True),
			nn.ReLU(),

			nn.Conv2d(256, 512, 4, 2, 1, bias=True),
			nn.InstanceNorm2d(512, affine=True),
			nn.ReLU(),

			nn.Conv2d(512, self.output_dim, 4, 2, 1, bias=True),
			nn.Sigmoid(),
		)

		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv(input)
		return x



class EEG_Encoder(object):
	def __init__(self, args):
		#parameters
		self.batch_size = args.batch_size
		self.epoch = args.epoch
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = 'EEG_ImageNet'#args.dataset
		self.dataroot_dir = '../../eegImagenet/mindbigdata-imagenet-in-v1.0/MindBigData-Imagenet-v1.0-Imgs'
		self.model_name = args.gan_type + args.comment
		self.sample_num = 128 #args.sample_num
		self.gpu_mode = args.gpu_mode
		self.num_workers = args.num_workers
		self.beta1 = args.beta1
		self.beta2 = args.beta2
		self.lrG = args.lrG
		self.lrD = args.lrD
		self.lrE = args.lrD
		self.type = 'train'
		self.lambda_ = 0.25
		self.n_critic = args.n_critic

		#load dataset
		self.data_loader = DataLoader(utils.EEG_ImageNet(root_dir = self.dataroot_dir,transform=transforms.Compose([transforms.Scale(100), transforms.RandomCrop(64), transforms.ToTensor()]),_type = self.type), batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
		self.enc_dim = 300
		self.num_cls = self.data_loader.dataset.num_cls

		self.E = Encoder()

		self.E_optimizer = optim.Adam(self.E.parameters(), lr=self.lrE, betas=(self.beta1, self.beta2))


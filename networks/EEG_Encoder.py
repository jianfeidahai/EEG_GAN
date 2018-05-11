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


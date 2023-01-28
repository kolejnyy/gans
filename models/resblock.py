import torch
import numpy as np
from tqdm import tqdm
from os import listdir
from torch import nn, optim, functional as F
from time import time

class ResidualBlock(nn.Module):

	def __init__(self, hidden_dim):
		super().__init__()

		self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
		self.norm1 = nn.BatchNorm2d(hidden_dim)
		self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
		self.norm2 = nn.BatchNorm2d(hidden_dim)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		out = self.conv1(x)
		out = self.norm1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.norm2(out)
		return out + x
	

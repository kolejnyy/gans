import torch
import numpy as np
from tqdm import tqdm
from os import listdir
from torch import nn, optim, functional as F
from time import time
from .resblock import ResidualBlock
from PIL import Image

class CycleGenerator(nn.Module):

	def __init__(self, hidden_dim = 128, image_channels = 3):
		super().__init__()

		self.hidden_dim = hidden_dim
		self.image_channels = image_channels

		self.intro = nn.Sequential(
			nn.Conv2d(self.image_channels, self.hidden_dim//4, 7, 1, 3),
			nn.InstanceNorm2d(self.hidden_dim//4),
			nn.ReLU(inplace=True)
		)

		self.down1 = nn.Sequential(
			nn.Conv2d(self.hidden_dim//4, self.hidden_dim//2, 3, 2, 1),
			nn.InstanceNorm2d(self.hidden_dim//2),
			nn.ReLU(inplace=True)
		)

		self.down2 = nn.Sequential(
			nn.Conv2d(self.hidden_dim//2, self.hidden_dim, 3, 2, 1),
			nn.InstanceNorm2d(self.hidden_dim),
			nn.ReLU(inplace=True)
		)

		self.resblocks = nn.Sequential(
			ResidualBlock(self.hidden_dim),
			ResidualBlock(self.hidden_dim),
			ResidualBlock(self.hidden_dim),
			ResidualBlock(self.hidden_dim)
		)

		self.up1 = nn.Sequential(
			nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim//2, 3, 2, 1, output_padding=1),
			nn.InstanceNorm2d(self.hidden_dim//2),
			nn.ReLU(inplace=True)
		)
		
		self.up2 = nn.Sequential(
			nn.ConvTranspose2d(self.hidden_dim//2, self.hidden_dim//4, 3, 2, 1, output_padding=1),
			nn.InstanceNorm2d(self.hidden_dim//4),
			nn.ReLU(inplace=True)
		)

		self.outro = nn.Sequential(
			nn.Conv2d(self.hidden_dim//4, self.image_channels, 7, 1, 3),
			nn.Sigmoid()
		)

	def forward(self, x):
		
		x = self.intro(x)
		x = self.down1(x)
		x = self.down2(x)
		x = self.resblocks(x)
		x = self.up1(x)
		x = self.up2(x)
		x = self.outro(x)

		return x


class PatchGAN(nn.Module):

	def __init__(self, image_channels = 1, hidden_dim = 128) -> None:
		super().__init__()


		self.layer1 = nn.Sequential(
			nn.Conv2d(image_channels, hidden_dim//2, 4, 2, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(hidden_dim//2, hidden_dim, 4, 2, 1, bias=False),
			nn.BatchNorm2d(hidden_dim),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer3 = nn.Sequential(
			nn.Conv2d(hidden_dim, 2*hidden_dim, 4, 2, 1, bias=False),
			nn.BatchNorm2d(2*hidden_dim),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer4 = nn.Sequential(
			nn.Conv2d(2*hidden_dim, 4*hidden_dim, 4, 2, 1, bias=False),
			nn.BatchNorm2d(4*hidden_dim),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer5 = nn.Sequential(
			nn.Conv2d(4*hidden_dim, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		return x.view(-1, 1).squeeze(1)



class CycleDiscriminator(nn.Module):

	def __init__(self, hidden_dim = 128, image_channels = 1) -> None:
		super().__init__()
		self.size = 70
		self.step = 20
		self.patchgan = PatchGAN(image_channels, hidden_dim)

	def forward(self, x):
		
		results = [self.patchgan(x[:, :, i:i+self.size, j:j+self.size])
					for i in range(0, 256-self.size, self.step) for j in range(0, 256-self.size, self.step)]
		return torch.mean(torch.stack(results))


class CycleGAN(nn.Module):

	def __init__(self, image_channels=3, hidden_dim=128, device = 'cpu'):
		super(CycleGAN, self).__init__()
		self.device = torch.device(device)

		self.gen_1 = CycleGenerator(hidden_dim, image_channels).to(self.device)
		self.gen_2 = CycleGenerator(hidden_dim, image_channels).to(self.device)
		self.disc_1 = CycleDiscriminator(hidden_dim, image_channels).to(self.device)
		self.disc_2 = CycleDiscriminator(hidden_dim, image_channels).to(self.device)

		self.disc_1_optim 	= optim.Adam(self.disc_1.parameters(), lr=0.0002, betas=(0.5, 0.999))
		self.disc_2_optim 	= optim.Adam(self.disc_2.parameters(), lr=0.0002, betas=(0.5, 0.999))
		self.gen_1_optim 	= optim.Adam(self.gen_1.parameters(), lr=0.0002, betas=(0.5, 0.999))
		self.gen_2_optim 	= optim.Adam(self.gen_2.parameters(), lr=0.0002, betas=(0.5, 0.999))

		self.loss = nn.MSELoss()
		self.cycle_loss = nn.L1Loss()

		self.lmbd = 10

	def forward(self, x, direction = True):
		if direction:
			return self.gen_1(x)
		return self.gen_2(x)

	
	def save(self, path):
		torch.save(self.state_dict(), path)
	
	def load(self, path):
		self.load_state_dict(torch.load(path))

	def generate_from_file(self, image_path, direction = True, save_real_path = None, save_fake_path = None):
		real_img = Image.open(image_path)
		if save_real_path:
			real_img.save(save_real_path)

		img_data = torch.from_numpy(np.array([np.array(real_img)])).float()/255
		img_data = img_data.permute(0, 3, 1, 2)
		img_data = img_data.to(self.device)
		with torch.no_grad():
			generated = None
			if direction:
				generated = self.gen_1(img_data)
			else:
				generated = self.gen_2(img_data)

		generated = generated.cpu().detach().numpy()[0]
		img_data = np.transpose(generated, (1, 2, 0))*255
		img_data = Image.fromarray(np.uint8(img_data))
		if save_fake_path:
			img_data.save(save_fake_path)
		return img_data

	def train(self, epochs, data_path_1, data_path_2, save_path):
		print('\n\n', "==================== Loading Data ====================")
		
		filenames_1 = listdir(data_path_1)
		filenames_2 = listdir(data_path_2)[:3000]
		
		dataset_1 = torch.from_numpy(np.array([[np.array(Image.open(data_path_1 + filename))] for filename in tqdm(filenames_1)])).float()/255
		dataset_2 = torch.from_numpy(np.array([[np.array(Image.open(data_path_2 + filename))] for filename in tqdm(filenames_2)])).float()/255

		dataset_1 = dataset_1.permute(0, 1, 4, 2, 3)
		dataset_2 = dataset_2.permute(0, 1, 4, 2, 3)

		batches_num = min(dataset_1.size()[0], dataset_2.size()[0])

		print("==================== Training ====================")
		
		for epoch in range(epochs):
			
			perm_1 = torch.randperm(dataset_1.size()[0])
			perm_2 = torch.randperm(dataset_2.size()[0])

			disc_avg_loss = 0
			gen_avg_loss = 0

			for i in tqdm(range(batches_num)):

				# ===================== Zero Gradients =====================

				self.disc_1.zero_grad()
				self.disc_2.zero_grad()
				self.gen_1.zero_grad()
				self.gen_2.zero_grad()

				# ===================== Train Discriminator =====================
				x = dataset_1[perm_1[i]].to(self.device)
				y = dataset_2[perm_2[i]].to(self.device)

				x_fake = self.gen_1(x)
				y_fake = self.gen_2(y)

				x_fake_disc = self.disc_1(x_fake.cpu().detach().to(self.device))
				y_fake_disc = self.disc_2(y_fake.cpu().detach().to(self.device))

				x_disc = self.disc_2(x)
				y_disc = self.disc_1(y)

				x_fake_loss = self.loss(x_fake_disc, torch.zeros_like(x_fake_disc))
				y_fake_loss = self.loss(y_fake_disc, torch.zeros_like(y_fake_disc))
				x_loss = self.loss(x_disc, torch.ones_like(x_disc))
				y_loss = self.loss(y_disc, torch.ones_like(y_disc))

				disc_loss = x_fake_loss + y_fake_loss + x_loss + y_loss
				
				self.disc_1_optim.zero_grad()
				self.disc_2_optim.zero_grad()
				
				disc_loss.backward()

				self.disc_1_optim.step()
				self.disc_2_optim.step()
				
				# ===================== Train Generator =====================

				x_fake_gen = self.disc_1(x_fake)
				y_fake_gen = self.disc_2(y_fake)
				
				cycle_loss = self.cycle_loss(self.gen_2(x_fake), x) + self.cycle_loss(self.gen_1(y_fake), y)
				gen1_loss = self.loss(x_fake_gen, torch.ones_like(x_fake_gen))
				gen2_loss = self.loss(y_fake_gen, torch.ones_like(y_fake_gen))

				gen_loss = gen1_loss + gen2_loss + self.lmbd*cycle_loss

				self.gen_1_optim.zero_grad()
				self.gen_2_optim.zero_grad()

				gen_loss.backward()

				self.gen_1_optim.step()
				self.gen_2_optim.step()

				# ===================== Update Losses =====================

				disc_avg_loss += disc_loss.item()
				gen_avg_loss += gen_loss.item()

				# ===================== Draw Examples =====================

				if i % 100 == 0:
					self.generate_from_file('data/monet2photo/testA/00010.jpg', True, 'debug/real_1.jpg', 'debug/fake_1.jpg')
					self.generate_from_file('data/monet2photo/testA/00020.jpg', True, 'debug/real_2.jpg', 'debug/fake_2.jpg')
					self.generate_from_file('data/monet2photo/testB/2014-08-02 15_56_41.jpg', False, 'debug/real_3.jpg', 'debug/fake_3.jpg')
					self.generate_from_file('data/monet2photo/testB/2014-08-04 20_20_12.jpg', False, 'debug/real_4.jpg', 'debug/fake_4.jpg')

			print("Epoch: ", epoch, "Disc Loss: ", disc_avg_loss/batches_num, "Gen Loss: ", gen_avg_loss/batches_num)
			self.save(save_path+'epoch_'+str(epoch)+'.pth')

		print("Training finished!\n\n")
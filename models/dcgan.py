import torch
import numpy as np
from tqdm import tqdm
from os import listdir
from torch import nn, optim, functional as F
from time import time

class Generator(nn.Module):

	def __init__(self, input_size = 100, image_channels = 1, f_num = 128):
		super().__init__()

		self.input_size = input_size
		self.f_num = f_num
		self.image_channels = image_channels

		self.layer1 = nn.Sequential(
			nn.ConvTranspose2d(self.input_size, self.f_num * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.f_num * 8),
			nn.ReLU(inplace=True)
		)

		self.layer2 = nn.Sequential(
			nn.ConvTranspose2d(self.f_num * 8, self.f_num * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.f_num * 4),
			nn.ReLU(inplace=True)
		)

		self.layer3 = nn.Sequential(
			nn.ConvTranspose2d(self.f_num * 4, self.f_num * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.f_num * 2),
			nn.ReLU(inplace=True)
		)

		self.layer4 = nn.Sequential(
			nn.ConvTranspose2d(self.f_num * 2, self.f_num, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.f_num),
			nn.ReLU(inplace=True)
		)

		self.layer5 = nn.Sequential(
			nn.ConvTranspose2d(self.f_num, image_channels, 4, 2, 1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = x.view(x.size(0), self.input_size, 1, 1)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		return x


class Discriminator(nn.Module):

	def __init__(self, image_channels = 1, hidden_dim = 128) -> None:
		super().__init__()

		self.image_channels = image_channels
		self.hidden_dim = hidden_dim

		self.layer1 = nn.Sequential(
			nn.Conv2d(self.image_channels, self.hidden_dim, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(self.hidden_dim, self.hidden_dim * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.hidden_dim * 2),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer3 = nn.Sequential(
			nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.hidden_dim * 4),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer4 = nn.Sequential(
			nn.Conv2d(self.hidden_dim * 4, self.hidden_dim * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.hidden_dim * 8),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer5 = nn.Sequential(
			nn.Conv2d(self.hidden_dim * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		return x.view(-1, 1).squeeze(1)



class DCGAN(nn.Module):

	def __init__(self, z_dim, image_channels, hidden_dim=128, device = 'cpu'):
		super(DCGAN, self).__init__()
		self.device = torch.device(device)

		self.z_dim = z_dim
		self.gen = Generator(z_dim, image_channels, hidden_dim).to(self.device)
		self.disc = Discriminator(image_channels, hidden_dim).to(self.device)
		
		self.disc_optim = optim.Adam(self.disc.parameters(), lr=0.0002, betas=(0.5, 0.999))
		self.gen_optim 	= optim.Adam(self.gen.parameters(), lr=0.0002, betas=(0.5, 0.999))

		self.loss = nn.BCELoss()


	def forward(self, x):
		return self.gen(x)
	
	def sample(self, n):
		z = torch.randn(n, self.z_dim).to(self.device)
		return self.gen(z)
	
	def save(self, path):
		torch.save(self.state_dict(), path)
	
	def load(self, path):
		self.load_state_dict(torch.load(path))

	def train(self, epochs, batch_size, data_path, disc_step, gen_step, save_path, save_step):
		print('\n\n', "==================== Loading Data ====================")
		dataset = np.array([np.load(data_path + file) for file in tqdm(listdir(data_path))])
		print("Dataset shape: ", dataset.shape, '\n')

		print("==================== Training ====================")
		
		for epoch in range(epochs):
			
			# Prepare batches
			np.random.shuffle(dataset)
			batches = [dataset[i:i+batch_size] for i in range(0, dataset.shape[0]-batch_size, batch_size)]
			
			start_time = time()

			d_loss = []
			g_loss = []

			for batch in tqdm(batches):

				real_labels = torch.ones(batch_size).to(self.device)
				fake_labels = torch.zeros(batch_size).to(self.device)

				# Train discriminator
				for i in range(disc_step):
					# Reset the gradient
					self.disc_optim.zero_grad()
					self.disc.zero_grad()

					# Calculate the loss on real data
					real = torch.from_numpy(batch[:,None,:,:]).to(self.device).float()
					real_loss = self.loss(self.disc(real), real_labels)
					real_loss = torch.mean(real_loss)

					# Calculate the loss on fake data
					z = torch.randn(batch_size, self.z_dim).to(self.device)

					with torch.no_grad():
						fake = self.gen(z)
					fake_loss = self.loss(self.disc(fake), fake_labels)
					fake_loss = torch.mean(fake_loss)

					disc_loss = real_loss + fake_loss
					d_loss.append(disc_loss.item())
					disc_loss.backward()
					self.disc_optim.step()
				
				# Train generator
				for i in range(gen_step):
					
					# Reset the gradient
					self.gen_optim.zero_grad()
					self.gen.zero_grad()

					# Calculate the loss on fake data
					z = torch.randn(batch_size, self.z_dim).to(self.device)
					fake = self.gen(z)
					gen_loss = self.loss(self.disc(fake), real_labels)
					gen_loss = torch.mean(gen_loss)

					g_loss.append(gen_loss.item())
					gen_loss.backward()
					self.gen_optim.step()

			print("Epoch: ", epoch, '/', epochs, "\td_loss: ", np.mean(d_loss), "\tg_loss: ", np.mean(g_loss), '\ttime:', time() - start_time, '\n')

			if epoch % save_step == save_step-1:
				self.save(save_path + 'epoch_' + str(epoch) + '.pth')
				print("Model saved at epoch: ", epoch, '\n')
			
		print("Training finished!\n\n")
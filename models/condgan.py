import cv2
import torch
import numpy as np
from tqdm import tqdm
from os import listdir
from torch import nn, optim, functional as F
from time import time

class CondGenerator(nn.Module):

	def __init__(self, input_size = 100, cond_size = 10, image_channels = 1, hidden_dim = 128):
		super().__init__()

		self.input_size = input_size
		self.context_dim = cond_size
		self.hidden_dim = hidden_dim
		self.image_channels = image_channels

		self.layer1_noise = nn.Sequential(
			nn.ConvTranspose2d(self.input_size, self.hidden_dim * 4, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.hidden_dim * 4),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer1_cond = nn.Sequential(
			nn.ConvTranspose2d(self.context_dim, self.hidden_dim * 4, 4, 1, 0, bias=False),
			nn.BatchNorm2d(self.hidden_dim * 4),
			nn.LeakyReLU(0.2, inplace=True),
		)	

		self.layer2 = nn.Sequential(
			nn.ConvTranspose2d(self.hidden_dim * 8, self.hidden_dim * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.hidden_dim * 4),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer3 = nn.Sequential(
			nn.ConvTranspose2d(self.hidden_dim * 4, self.hidden_dim * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.hidden_dim * 2),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer4 = nn.Sequential(
			nn.ConvTranspose2d(self.hidden_dim * 2, self.hidden_dim, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.hidden_dim),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer5 = nn.Sequential(
			nn.ConvTranspose2d(self.hidden_dim, image_channels, 4, 2, 1, bias=False),
			nn.Sigmoid()
		)

	def forward(self, noise, context):
		x = noise.view(noise.size(0), self.input_size, 1, 1)
		y = context.view(context.size(0), self.context_dim, 1, 1)

		x = self.layer1_noise(x)
		y = self.layer1_cond(y)
		
		x = torch.cat([x, y], dim=1)

		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		return x


class CondDiscriminator(nn.Module):

	def __init__(self, image_channels = 1, hidden_dim = 128, final_dim = 200, context_dim = 10) -> None:
		super().__init__()

		self.image_channels = image_channels
		self.hidden_dim = hidden_dim
		self.final_dim = final_dim
		self.context_dim = context_dim

		self.layer1_noise = nn.Sequential(
			nn.Conv2d(self.image_channels, self.hidden_dim, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer1_cond = nn.Sequential(
			nn.Conv2d(self.context_dim, self.hidden_dim, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(self.hidden_dim*2, self.hidden_dim * 2, 4, 2, 1, bias=False),
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

	def forward(self, noise, context):
		
		batch_size = noise.size(0)

		context = context.view(batch_size, self.context_dim, 1, 1)
		context = context.expand((batch_size, self.context_dim, noise.size(2), noise.size(3)))

		x = self.layer1_noise(noise)
		y = self.layer1_cond(context)
		x = torch.cat([x, y], dim=1)

		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.layer5(x)
		return x.view(-1, 1).squeeze(1)



class CondGAN(nn.Module):

	def __init__(self, z_dim, cond_dim, image_channels, hidden_dim=128, final_dim = 200, device = 'cpu'):
		super(CondGAN, self).__init__()
		self.device = torch.device(device)

		self.z_dim = z_dim
		self.cond_dim = cond_dim
		self.gen = CondGenerator(z_dim, cond_dim, image_channels, hidden_dim).to(self.device)
		self.disc = CondDiscriminator(image_channels, hidden_dim, final_dim=final_dim, context_dim=cond_dim).to(self.device)
		
		self.disc_optim = optim.Adam(self.disc.parameters(), lr=0.0001, betas=(0.5, 0.999))
		self.gen_optim 	= optim.Adam(self.gen.parameters(), lr=0.0001, betas=(0.5, 0.999))

		self.loss = nn.BCELoss()


	def forward(self, x, context):
		return self.gen(x, context)
	
	def sample(self, n, context):
		z = torch.randn(n, self.z_dim).to(self.device)
		return self.gen(z, context)
	
	def save(self, path):
		torch.save(self.state_dict(), path)
	
	def load(self, path):
		self.load_state_dict(torch.load(path))

	def train(self, epochs, batch_size, data_path, disc_step, gen_step, save_path, save_step, draw_step=None):
		
		print('\n\n', "==================== Loading Data ====================")
		
		filenames = listdir(data_path)
		filenames.sort()
		id = np.identity(self.cond_dim)
		dataset = np.array([np.load(data_path + file).astype(np.float)/255 for file in tqdm(filenames)])
		labels = np.array([id[int(file.split('_')[0])] for file in tqdm(filenames)])
		data_size = dataset.shape[0]
		print("Dataset size: ", data_size, '\n')

		
		
		print("==================== Training ====================")
		
		real_labels = torch.ones(batch_size).to(self.device)
		fake_labels = torch.zeros(batch_size).to(self.device)

		for epoch in range(epochs):
			
			# Shuffle data
			perm = np.random.permutation(dataset.shape[0])
			dataset = dataset[perm]
			labels = labels[perm]
			
			# Prepare batches
			batches_x = np.array([dataset[i:i+batch_size] for i in range(0, data_size-batch_size, batch_size)])
			batches_l = np.array([labels[i:i+batch_size] for i in range(0, data_size-batch_size, batch_size)])
			start_time = time()

			d_loss = []
			g_loss = []


			for i in tqdm(range(batches_x.shape[0])):

				batch = batches_x[i]
				batch_labels = batches_l[i]

				try:
					real = torch.from_numpy(batch[:,None,:,:]).to(self.device).float()
				except:
					print(torch.from_numpy(batch[:,None,:,:]).to(self.device).shape)
					print("Error in batch: ", i)
					return
				real_context = torch.from_numpy(batch_labels).to(self.device).float()

				# Train discriminator
				for disci in range(disc_step):
					# Reset the gradient
					self.disc_optim.zero_grad()
					self.disc.zero_grad()

					# Calculate the loss on real data
					real_loss = self.loss(self.disc(real, real_context), real_labels)
					real_loss = torch.mean(real_loss)

					# Calculate the loss on fake data

					z = torch.randn(batch_size, self.z_dim).to(self.device)
					with torch.no_grad():
						fake = self.gen(z, real_context)
					fake_loss = self.loss(self.disc(fake, real_context), fake_labels)
					fake_loss = torch.mean(fake_loss)

					disc_loss = real_loss + fake_loss
					d_loss.append(disc_loss.item())
					disc_loss.backward()
					self.disc_optim.step()
				
				# Train generator
				for gsi in range(gen_step):
					
					# Reset the gradient
					self.gen_optim.zero_grad()
					self.gen.zero_grad()

					# Calculate the loss on fake data
					z = torch.randn(batch_size, self.z_dim).to(self.device)
					context = torch.from_numpy(batch_labels).to(self.device).float()
					fake = self.gen(z, context)
					gen_loss = self.loss(self.disc(fake, context), real_labels)
					gen_loss = torch.mean(gen_loss)

					g_loss.append(gen_loss.item())
					gen_loss.backward()
					self.gen_optim.step()

				# If draw_step is specified, draw sample images
				if draw_step != None:
					if i%draw_step==0:
						with torch.no_grad():
							noise = torch.randn(self.cond_dim, self.z_dim).to('cuda:0')
							id_labels = torch.from_numpy(np.identity(self.cond_dim)).to('cuda:0').float()
							out = self.gen(noise, id_labels).detach().cpu().numpy()
							for j in range(self.cond_dim):
								cv2.imwrite('debug/test_{}.png'.format(j), out[j][0]*255)


			print("Epoch: ", epoch, '/', epochs, "\td_loss: ", np.mean(d_loss), "\tg_loss: ", np.mean(g_loss), '\ttime:', time() - start_time, '\n')

			if epoch % save_step == save_step-1:
				self.save(save_path + 'epoch_' + str(epoch) + '.pth')
				print("Model saved at epoch: ", epoch, '\n')
			
		print("Training finished!\n\n")
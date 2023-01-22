# Generative Adversarial Networks (GANs) in PyTorch

### TODO

- [ ] Create codes for all considered GANs:
  - [x] DCGAN [(arXiv)](https://arxiv.org/abs/1511.06434)
  - [x] CondGAN [(paper)](http://cs231n.stanford.edu/reports/2015/pdfs/jgauthie_final_report.pdf)
  - [ ] CycleGAN [(arXiv)](https://arxiv.org/abs/1703.10593)
- [ ] Train the relevant models on relevant datasets
- [ ] Generate a birthday gift for my girlfriend


## DCGAN

The DCGAN is a GAN that uses convolutional layers instead of fully connected layers. It was introduced in the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Alec Radford, Luke Metz, and Soumith Chintala. During the expreriment, I trained the model on the MNIST digit dataset. The model was trained for 10 epochs, using the training specification provided in the beforementioned paper and the results are shown below.

<p float = "left">
	<img src="results/dcgan_0.png" width = 9%>
	<img src="results/dcgan_1.png" width = 9%>
	<img src="results/dcgan_2.png" width = 9%>
	<img src="results/dcgan_3.png" width = 9%>
	<img src="results/dcgan_4.png" width = 9%>
	<img src="results/dcgan_5.png" width = 9%>
	<img src="results/dcgan_6.png" width = 9%>
	<img src="results/dcgan_7.png" width = 9%>
	<img src="results/dcgan_8.png" width = 9%>
	<img src="results/dcgan_9.png" width = 9%>
</p>


## CondGAN

Conditional Generative Adversarial Network (CondGAN) apart from an input noise, also takes a  class label as a part of the input, for both generator and discriminator. By this modification, the model is able to generate images of a specific class. The model was train on the MNIST dataset, and already after one epoch was able to generate digits with provided label. Over the next few epochs, the model's accuracy (assessed by hand) started to decay, most likely due to overfitting of the discriminator. The results are shown below:

<img src="results/condgan.png" width = 100%>
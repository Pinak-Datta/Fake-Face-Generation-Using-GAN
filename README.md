# Fake-Face-Generation-Using-GAN
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This project is a Python implementation of the Deep Convolutional Generative Adversarial Network (DCGAN) for generating realistic fake faces using PyTorch. The network architecture used in this project is inspired by the paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Alec Radford, Luke Metz, and Soumith Chintala.

## Dependencies
Python 3.6+  
PyTorch 1.9+  
TorchVision 0.10+  
Matplotlib  

## Results
![Output for 1000 Epochs](https://github.com/Pinak-Datta/Fake-Face-Generation-Using-GAN/blob/main/output_image.jpg)
![Another Generated Output for 1000 Epochs](https://github.com/Pinak-Datta/Fake-Face-Generation-Using-GAN/blob/main/output_image_1000.jpg)

## Installation
1. Clone the repository
```
git clone https://github.com/Pinak-Datta/Fake-Face-Generation-Using-GAN.git
```  
2. Install the dependencies using pip:
```
pip install -r requirements.txt
``` 

## Usage
To generate fake faces using the pre-trained model, run the following command: (Note that here you need to supply the path to the image you want to create a fake of)  
```
python generate.py
```
To train the model on your own dataset, place your images in the data directory. To increase and decrease the number of epochs, change the num_epochs and run the following command:
```
python Train.py
```

## Acknowledgements
This project is inspired by the work of Alec Radford, Luke Metz, and Soumith Chintala on DCGAN. The pre-trained model used in this project is based on the [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) dataset.

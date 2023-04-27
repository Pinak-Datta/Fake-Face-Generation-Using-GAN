# Fake-Face-Generation-Using-GAN

This project is a Python implementation of the Deep Convolutional Generative Adversarial Network (DCGAN) for generating realistic fake faces using PyTorch. The network architecture used in this project is inspired by the paper "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" by Alec Radford, Luke Metz, and Soumith Chintala.

## Dependencies
Python 3.6+  
PyTorch 1.9+  
TorchVision 0.10+  
Matplotlib  

## Results

## Installation
1. Clone the repository
```
git clone https://github.com/username/repo_name.git
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

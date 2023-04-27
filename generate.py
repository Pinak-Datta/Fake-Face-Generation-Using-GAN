import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# Define the generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


# Load the saved model
netG = Generator()
netG.load_state_dict(torch.load("C:\\Users\\KIIT\\PycharmProjects\\Minor\\netG.pth"))

# Set the device to use
device = torch.device("cpu")
netG.to(device)

# Define the transform to resize the input image to the same size as the training images
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load the input image
input_image = Image.open("C:\\Users\\KIIT\\PycharmProjects\\TTL_28_Mar\\my_photo.jpg")
input_tensor = transform(input_image).unsqueeze(0).to(device)

# Generate the output image
with torch.no_grad():
    noise = torch.randn(1, 100, 1, 1, device=device)
    output_tensor = netG(noise)
output_image = transforms.ToPILImage()(output_tensor.squeeze().cpu())

# Save the generated image
output_image.save("C:\\Users\\KIIT\\PycharmProjects\\Minor\\output_image_1000.jpg")

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        self.pool2 = nn.MaxPool2d(2,2)
        self.pool4 = nn.MaxPool2d(4,4)
        # First conv layers
        self.conv1 = nn.Conv2d(3, 64, 7, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256,512, 7, padding=3)
        #self.conv5 = nn.Conv2d(12,6, 3, padding=1)
        #self.conv6 = nn.Conv2d(6,3, 1, padding=0)
        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 12)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool4(x)
        x = F.relu(self.conv4(x))
        
        x = self.pool4(x)
        #x = F.relu(self.conv5(x))
        #x = F.relu(self.conv6(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# initialize the NN
model = ConvAutoencoder()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

from PIL import Image
from torchvision.transforms import ToTensor

mapping = {'afil_azul': 0,
 'afil_blanco': 1,
 'caballo_azul': 2,
 'caballo_blanco': 3,
 'peon_azul': 4,
 'peon_blanco': 5,
 'reina_azul': 6,
 'reina_blanco': 7,
 'rey_azul': 8,
 'rey_blanco': 9,
 'torre_azul': 10,
 'torre_blanco': 11}

rev_mapping = {v:k for k,v in mapping.items()}

im = Image.open('knight.jpg')
im = im.resize((256, 256))
im = ToTensor()(im).unsqueeze(0)

y = model(im)
y = torch.max(y, 1)

for idx in y.indices.numpy():
    print(rev_mapping[idx])

def infer(data):
    pass
from django.shortcuts import render
from .forms import AnimalForm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor
import base64
import os
from PIL import Image 

PATH = csv_path = os.path.join(os.path.dirname(__file__), 'state_dict_model.pt')
Labels = {0: 'Cheetah', 1: 'Fox', 2: 'Hyena', 3: 'Lion', 4: 'Tiger',5:'Wolf'}
trans = torchvision.transforms.Compose([
    torchvision.transforms.RandAugment(),
    torchvision.transforms.AutoAugment(),
#     torchvision.transforms.RandomHorizontalFlip(p=0.5)
#     torchvision.transforms.RandomEqualize(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
#     torchvision.transforms.RandomInvert(p=0.5),
    torchvision.transforms.Resize(size =(300,300)),
    ToTensor()
])
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #Feature extraction
        self.conv= nn.Conv2d(3,64,9)
        self.conv2 = nn.Conv2d(64,128,5)
        self.conv3 = nn.Conv2d(128,256,5)
        self.conv4 = nn.Conv2d(256,512,5)
        self.conv5 = nn.Conv2d(512,1024,5)
        self.bat2d1 = nn.BatchNorm2d(64)
        self.bat2d2 = nn.BatchNorm2d(128)
        self.bat2d3 = nn.BatchNorm2d(256)
        self.bat2d4 = nn.BatchNorm2d(512)
        self.bat2d5 = nn.BatchNorm2d(1024)
        self.bat1d = nn.BatchNorm1d(196)
        self.bat1d2 = nn.BatchNorm1d(256)
#         self.bat1d3 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p= 0.8)
        self.dropout2 = nn.Dropout(p=0.5)

        
        #Classification layer
        self.lin = nn.Linear(1024*5*5,196)
        self.lin3 = nn.Linear(196,256)
#         self.lin4 = nn.Linear(256,512)
        self.lin2 = nn.Linear(256,6)
        
    def forward(self,x):
        #First two lines are referenced
        x = F.max_pool2d(F.relu(self.conv(x)),(2,2))
        x=self.bat2d1(x)
        self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=self.bat2d2(x)
        self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        x=self.bat2d3(x)
        self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv4(x)),(2,2))
        x=self.bat2d4(x)
        self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv5(x)),(2,2))
        x=self.bat2d5(x)
        self.dropout(x)
#         print(x.shape)
        #Flattening tensors
        x = torch.flatten(x,1)
#         print(x.shape)
        x = F.relu(self.lin(x))
        self.bat1d(x)
        x= F.relu(self.lin3(x))
        self.bat1d2(x)
        self.dropout2(x)
#         x= F.relu(self.lin4(x))
#         self.bat1d3(x)
#         self.dropout2(x)
        x = F.softmax(self.lin2(x),dim = 1)
        return x
    
model = Model()
parallel_model = torch.nn.DataParallel(model)
parallel_model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
parallel_model.eval()
def index(request):
    image_uri = None
    predictedlabel = -1

    if request.method == 'POST':
        form = AnimalForm(request.POST,request.FILES)
        if form.is_valid():

            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)
            print(type(image))
            # Transforming image
            image = Image.open(image)
            itest = trans(image)
            itest = itest.unsqueeze(0)
            #Predicting image
            output = parallel_model(itest)
            maxele,maxindx = torch.max(output,1)
            predictedlabel = Labels[maxindx.item()]

            context = {'form': form, 'predict': predictedlabel,'image_uri': image_uri}
    else:
        form = AnimalForm()
        context = {'form': form, 'predict': predictedlabel,'image_uri': image_uri}
    if predictedlabel == 'Fox':
        return render(request,'classifier/fox.html',context)
    elif predictedlabel == 'Hyena':
        return render(request,'classifier/hyena.html',context)
    elif predictedlabel == 'Tiger':
        return render(request,'classifier/tiger.html',context)
    elif predictedlabel == 'Wolf':
        return render(request,'classifier/wolf.html',context)

    return render(request,'classifier/index.html',context)


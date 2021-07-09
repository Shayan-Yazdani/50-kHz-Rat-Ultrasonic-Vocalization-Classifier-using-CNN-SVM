
from torch.autograd import Variable
from torchvision import models
import torch
import torchvision.transforms as transforms
from Net import DenoisingAutoencoder
import pickle
from SpecMake import MakeSpec
import numpy as np
from skimage.transform import resize
from PIL import Image
import pandas as pd


def Classify(Fs, x,t):


    #Load alexnet
    AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    AlexNet_conv = torch.nn.Sequential(*list(AlexNet_model.children())[0:-1])
    scaler = transforms.Scale((224, 224))
    AlexNet_conv = torch.nn.Sequential(*list(AlexNet_model.children())[0:-1])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()

    #Load demoising autoencoder
    DenAutoencoder=DenoisingAutoencoder()
    DenAutoencoder.load_weights('autoencoder_weights.h5')

    Output={}
    cats=['Complex','Down ramp','Flat','Flat-trill combination','Inverted U','Multi','Step down','Short','Split','Step up','Trill with Jumps','Trill','Up ramp']
    Categories=[]
    Category=[]
    Spectrograms={}

    for j in range(0,len(t['Begin Time (s)'])):


        if   t['High Freq (kHz)'][j]<t['Low Freq (kHz)'][j]:
           continue

        T1=t['Begin Time (s)'][j]*Fs
        T2=t['End Time (s)'][j]*Fs
        lf = t['Low Freq (kHz)'][j]*1000
        hf = t['High Freq (kHz)'][j]*1000

        im=MakeSpec(x[int(T1):int(T2)],lf,hf,Fs)
        im=np.array(im)
        im2=resize(im, (224, 224))
        im2=im2.reshape(1, 224, 224, 1)
        imden = DenAutoencoder.predict(im2)
        img=abs(imden[0,:,:,0]*255).astype('uint8')
        img = np.stack((img,)*3, axis=-1)
        img=Image.fromarray(img, 'RGB')
        ximg = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
        outputs = AlexNet_conv(ximg)
        features=torch.flatten(outputs).detach().numpy()

        #Load  SVM model
        svm_model = pickle.load(open('SVMmodel.sav', 'rb'))

        #Predict the category
        features = np.expand_dims(features, axis=0)
        prediction=svm_model.predict(features)
        Probs=svm_model.predict_proba(features)
        Prob=np.max(Probs,axis=1)
        if Prob<0.4:
            Category="Unknown"
        else:
            Category=cats[prediction[0]-1]

        Categories.append(Category)
        Spectrograms[j]=im

    return Categories, Spectrograms

import numpy as np
from PIL import Image
import matplotlib.mlab



def MakeSpec(Audio,lf,hf,Fs):

    img=imageconv(Audio,Fs,int(lf),int(hf-lf))
    #im = Image.fromarray(img)
    return img


def imageconv(audio,Fs,fl,fh):

    S, fr, ti = matplotlib.mlab.specgram(audio, NFFT=256, Fs=Fs, noverlap=220, pad_to=1367)
    y1 = freq2idx(fr,fl)
    y2= freq2idx(fr,fh)+y1
    I=(S[y1:y2,0:-1])
    med = np.median(abs(np.matrix.flatten(I)))
    img = mat2gray(np.flipud(I),0, med*15)
    img=imrsz(img)
    img=img.astype('uint8')

    return img


def freq2idx(fr, freq):

    lng=len(fr)
    df=lng/(fr[-1]-fr[0])
    fidx=df*(freq-fr[0])
    return int(np.round(fidx))


def mat2gray(mat,low, high):

    mat[mat>high]=high
    matscld=mat/high

    return matscld*255


def imrsz(img):

    if np.shape(img)[0]<224:
        a=(224-np.shape(img)[0])//2
        b=(224-np.shape(img)[0])//2
        npad = ((a,b), (0, 0))
        img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)

    if np.shape(img)[1]<224:
        a=(224-np.shape(img)[1])//2
        b=(224-np.shape(img)[1])//2
        npad = ((0,0),(a,b))
        img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)

    return img

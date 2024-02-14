from torchvision import transforms 
import torchaudio
import torch
import numpy as np
import time
from torch.nn import functional as F

import torch 
import numpy as np 
import random 
import os,sys,json,glob
import soundfile
import numpy as np 
import soundfile
import random
import numpy as np
import torch
def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
    feat = np.stack(feats,axis=0).astype(np.float)
    return feat

class CustomAudioTransform:
    def __repr__(self):
        return self.__class__.__name__ + '()'
class Identity(CustomAudioTransform):
    def __call__(self,signal):
        return signal
    
class GaussianNoise(CustomAudioTransform):
    def __init__(self,g):
        self.g = g
    def __call__(self,signal):
        return signal + self.g * torch.randn_like(signal)

class PadToSize(CustomAudioTransform):
    def __init__(self, size:int):
        self.size = size

    def __call__(self, signal):

        if signal.shape[1] < self.size :
            signal = F.pad(signal, (0, self.size-signal.shape[1]))
        return signal

class ToSizeN(CustomAudioTransform):
    def __init__(self, size:int):
        self.size = size

    def __call__(self, signal):
        n = signal.shape[1]//self.size
        m = signal.shape[1] % self.size
        if m > self.size//2 or n==0:
            signal = F.pad(signal, (0, self.size*(n+1)-signal.shape[1]))
        else:
            signal = F.pad(signal, (0, self.size*n-signal.shape[1]))
        return signal

class CentralCrop(CustomAudioTransform):
    def __init__(self, size:int, pad:bool=True):
        self.size = size
        self.pad = pad

    def __call__(self, signal):

        if signal.shape[1] < self.size :
            if self.pad:
                signal = F.pad(signal, (0, self.size-signal.shape[1]))
            return signal

        start = (signal.shape[1] - self.size) // 2
        return signal[:, start: start + self.size]

class RandomCrop(CustomAudioTransform):
    def __init__(self, size:int, pad:bool = True):
        self.size = size
        self.pad = pad

    def __call__(self, signal):
        if signal.shape[1] < self.size :
            if self.pad:
                signal = F.pad(signal, (0, self.size-signal.shape[1]))
            return signal
        start = np.random.randint(0, signal.shape[-1] - self.size + 1)
        return signal[:, start: start + self.size]
    

class Normalize(CustomAudioTransform):
    def __init__(self,std_mean=None,reduce_dim=None):
        self.std_mean = std_mean
        self.reduce_dim = reduce_dim
    def __call__(self,input):
        """
        assuming input has shape [batch,nmels,time]
        """
        std,mean = None,None
        if self.std_mean is None:
            if self.reduce_dim is not None:
                std,mean = torch.std_mean(input,dim=self.reduce_dim,keepdim=True)
            else:
                std,mean = torch.std_mean(input)
        else:
            std,mean = self.std_mean
        output = input - mean 
        output = output / (std + 1e-6)
        return output

class MinMax(CustomAudioTransform):
    def __init__(self,min,max):
        self.min=min
        self.max=max
    def __call__(self,input):
        min_,max_ = None,None
        if self.min is None:
            min_ = torch.min(input)
            max_ = torch.max(input)
        else:
            min_ = self.min
            max_ = self.max
        input = (input - min_)/(max_- min_) *2. - 1.
        return input

class div(CustomAudioTransform):
    def __init__(self,value=100):
        self.value = value
    def __call__(self,input):
        input /= 100
        return input

class NoiseAug(CustomAudioTransform):

    def __init__(self, hparams):
        musan_path=hparams.musan_path
        rir_path = hparams.rir_path
        max_framesses=self.hparams.max_frames
        self.max_frames = max_frames
        self.max_audio  = max_audio = max_frames * 160 + 240
        self.noisetypes = ['noise','speech','music']

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}

        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'));

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 

        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        noises = []

        for noise in noiselist:

            noiseaudio  = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2)+1e-4) 
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        return numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True) + audio

    def reverberate(self, audio):

        rir_file    = random.choice(self.rir_files)
        
        rir, fs     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))

        return signal.convolve(audio, rir, mode='full')[:,:self.max_audio]

    def __call__(self, audio):
    
        augtype = random.randint(0,4)
        if augtype == 1:
            audio   = self.augment_wav.reverberate(audio)
        elif augtype == 2:
            audio   = self.augment_wav.additive_noise('music',audio)
        elif augtype == 3:
            audio   = self.augment_wav.additive_noise('speech',audio)
        elif augtype == 4:
            audio   = self.augment_wav.additive_noise('noise',audio)
        return audio

class RandomGain(CustomAudioTransform):
    pass
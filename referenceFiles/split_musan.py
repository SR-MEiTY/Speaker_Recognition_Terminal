import glob
from scipy.io import wavfile
import os
files = glob.glob('/home/iiitdwd/kaldiSpace/database/musan_all/*/*/*.wav')
audlen = 16000*5
audstr = 16000*3
for idx,file in enumerate(files):
    fs,aud = wavfile.read(file)
    writedir = os.path.splitext(file.replace('/musan_all/','/musan_split/'))[0]
    print(writedir)
    os.makedirs(writedir)
    for st in range(0,len(aud)-audlen,audstr):
        wavfile.write(writedir+'/%05d.wav'%(st/fs),fs,aud[st:st+audlen])
        #print(idx,file)

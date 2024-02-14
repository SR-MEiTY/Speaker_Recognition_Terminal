PATH_DATA= "/content/dataset/I-MSV-DATA/I-MSV-DATA/"
import tqdm
import librosa
import os,glob,sys
import soundfile as sf
from tqdm import tqdm
def read_and_split(xs):
  print(f"Process {len(xs)}")
  for path in tqdm(xs):
    data, samplerate = sf.read(path)
    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
    samplerate=16000
    
    sf.write(path, data, samplerate)
read_and_split(list(glob.glob(f"{PATH_DATA}/I_MSV_DEV_ENR/Enr_data/*.wav")))
read_and_split(list(glob.glob("split_ds/imsv-public-test/imsv-public-test/*.wav")))
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
#read_and_split(list(glob.glob("/home/iiitdwd/cocosda_wavlm/data/Enr_data-20230204T124515Z-001/Enr_data/*.wav")))
read_and_split(list(glob.glob("/home/iiitdwd/cocosda_wavlm/data/I-MSV-Public-test-20230204T124223Z-001/I-MSV-Public-test/data/*.wav")))

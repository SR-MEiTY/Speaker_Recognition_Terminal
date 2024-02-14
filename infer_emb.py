PATH_DATA_ROOT= "/content/unsplited/I-MSV-DATA/I-MSV-DATA"
PATH_PRIVATE_ROOT="/content/private_test"
PATH_PRIVATE_PANDAS_SUB_SAMPLE="/content/sub/results.csv"
PATH_CHECKPOINT='/content/drive/MyDrive/dataset/imsv/epoch=0-VEER=1.354-mindcf=0.056.ckpt'
import tqdm
import librosa
import os,glob,sys
import soundfile as sf
from tqdm import tqdm
from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity
from pytorch_lightning import LightningModule
# from audiossl.models.atst import ATST
from models.rawnet import MainModelRawnet
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from utils.common import cosine_scheduler_step,get_params_groups
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset.speaker_verification_data import SpeakerDataset, test_dataset_loader, get_train_transform, get_eval_transform
from dataset.sampler import HierarchicalSampler
import time
import torch
import warnings
warnings.simplefilter("ignore")
from callbacks.callbacks import ScoreCallback
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity
from utils.utils import evaluateFromList
from utils.tuneThreshold import *
from collections import OrderedDict
from dataset.speaker_verification_data import test_dataset_loader
import torch
import torch,os,glob,sys
from utils.tuneThreshold import tuneThresholdfromScore,ComputeMinDcf,ComputeErrorRates
# Define model 
from models.tdnn import ECAPA_TDNN_SMALL

from tqdm import tqdm
import sys
import logging
import argparse
import traceback
import pandas as pd

# Logger
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
def read_and_split(xs):
  print(f"Process {len(xs)}")
  for path in tqdm(xs):
    data, samplerate = sf.read(path)
    if samplerate==16000:continue
    data = librosa.resample(data, orig_sr=samplerate, target_sr=16000)
    samplerate=16000
    sf.write(path, data, samplerate)






# Resample to 16k + use enroll datasets without split 5s
read_and_split(list(glob.glob(f"{PATH_DATA_ROOT}/I_MSV_DEV_ENR/Enr_data/*.wav")))
read_and_split(list(glob.glob(f"{PATH_PRIVATE_ROOT}/*.wav")))

import pandas as pd
test=pd.read_csv(PATH_PRIVATE_PANDAS_SUB_SAMPLE)
speaker_id=test.groupby("utterance_id")['speaker_id'].apply(list).reset_index(name="spkears")
speaker_id['Utterance_ID']=["1067_",] * len(speaker_id)
speaker_id['rand_ID']=speaker_id['utterance_id']

a=torch.load(PATH_CHECKPOINT)['state_dict']
b={}
for k in a:
  if "model" not in k:continue
  b[k.replace("model.","")] = a[k]
model  = MainModelRawnet(
    nOut=256,
    sinc_stride=10,
    encoder_type="ECA"
)
print(model.load_state_dict(b))
testset = test_dataset_loader(
    list(glob.glob(f"{PATH_PRIVATE_ROOT}/*.wav")),
    "",
    500,
    num_eval=15   
)

test_loader = torch.utils.data.DataLoader(
    testset, 
    batch_size=1, shuffle=False,
    num_workers=2,
    drop_last=False
)

feats={}
model=model.cuda()
model.eval()
dc=[]
dx=[]
## Extract features for every image
for idx, data in tqdm(enumerate(test_loader),total=len(test_loader)):
    inp1 = data[0][0].cuda() #bs,10,num_audio
    dc.append(inp1)
    dx.append(data[1][0])
    if len(dc)==4 or idx==len(test_loader)-1:
      d_c=torch.stack(dc,0)# bs,5,num_audi
      nxx=d_c.size(1)
      bs=d_c.size(0)
      with torch.no_grad():
          ref_feat = model(d_c.reshape(bs * nxx,-1)).detach().cpu()
          ref_feat=ref_feat.reshape(bs,nxx,-1)# bs,-1
      for k,x in enumerate(dx):
        feats[x] = ref_feat[k]
      dx=[]
      d_c.to("cpu")
      dc=[]
    if idx % 1000==0 or idx == len(test_loader)-1:
      torch.save(feats, "test_emb_rawnet.pth")

    # feats[data[1][0]] = ref_feat

torch.save(feats, "test_emb.pth")


#------------------------------------------------------------------------------#


#Define cohort embedding



testset = test_dataset_loader(
    list(glob.glob(f"{PATH_DATA_ROOT}/I_MSV_DEV_ENR/Enr_data/*.wav")),
    "",
    500,
    num_eval=15   
)
test_loader = torch.utils.data.DataLoader(
    testset, 
    batch_size=1, shuffle=False,
    num_workers=2,
    drop_last=False
)

feats={}
model=model.cuda()
model.eval()
from tqdm import tqdm
## Extract features for every image
for idx, data in tqdm(enumerate(test_loader),total=len(test_loader)):
    inp1 = data[0][0].cuda() #bs,10,num_audio
    with torch.no_grad():
        ref_feat = model(inp1).detach().cpu()
    
    feats[data[1][0]] = ref_feat.mean(0).reshape(1,-1)


speaker_feat={}
list(feats.keys())[0]
for key in feats.keys():
  emb=feats[key]
  spk = os.path.basename(key).split("_")[0]
  spk=str(spk) 
  if spk not in speaker_feat:speaker_feat[spk] = []
  speaker_feat[spk].append(emb)
for key in speaker_feat.keys():
  speaker_feat[key] = torch.cat(speaker_feat[key],0)
# torch.save(speaker_feat, "/content/drive/MyDrive/dataset/imsv/private_predict/env_emb_rawnet.pth")
torch.save(speaker_feat,"env_emb.pth")


class Args:
  method="asnorm"
  top_n=300
  second_cohort="true"
  cross_select="true"

from tqdm import tqdm
import sys
import logging
import argparse
import traceback
import pandas as pd
def asnorm(args,input_score, enroll_cohort_score,test_cohort_score):
    enroll_test_names = ["enroll", "test", "score"]
    enroll_cohort_names = ["enroll", "cohort", "score"]
    test_cohort_names = ["test", "cohort", "score"]
    # input_score = load_score(args.input_score, enroll_test_names)
    # enroll_cohort_score = load_score(args.enroll_cohort_score, enroll_cohort_names)
    # test_cohort_score = load_score(args.test_cohort_score, test_cohort_names)

    output_score = []

    
    enroll_cohort_score.sort_values(by="score", ascending=False, inplace=True)
    test_cohort_score.sort_values(by="score", ascending=False, inplace=True)

    if args.cross_select == "true":
       
        enroll_top_n = enroll_cohort_score.groupby("enroll").head(args.top_n)[["enroll", "cohort"]]
        test_group = pd.merge(
            pd.merge(input_score[["enroll", "test"]], enroll_top_n, on="enroll"), 
            test_cohort_score, on=["test", "cohort"]).groupby(["enroll", "test"])

        test_top_n = test_cohort_score.groupby("test").head(args.top_n)[["test", "cohort"]]
        enroll_group = pd.merge(pd.merge(input_score[["enroll", "test"]], test_top_n, on="test"), 
                                enroll_cohort_score, on=["enroll", "cohort"]).groupby(["enroll", "test"])
    else:
        enroll_group = enroll_cohort_score.groupby("enroll").head(args.top_n).groupby("enroll")
        test_group = test_cohort_score.groupby("test").head(args.top_n).groupby("test")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    for _, row in input_score.iterrows():
        enroll_key, test_key, score = row
        if args.cross_select == "true":
            normed_score = 0.5 * ((score - enroll_mean[enroll_key, test_key]) / enroll_std[enroll_key, test_key] + \
                                 (score - test_mean[enroll_key, test_key]) / test_std[enroll_key, test_key])
        else:
            normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] + \
                                (score - test_mean[test_key]) / test_std[test_key])
        output_score.append([enroll_key, test_key, normed_score])

    return output_score

from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity
speaker_feat=torch.load("env_emb.pth")
test_feat=torch.load("test_emb.pth")
e=[]
for k in speaker_feat.keys():
  e.extend([k,] )
cohort = torch.cat([speaker_feat[ci] for ci in speaker_feat.keys()],0)
dsx=[]
enroll_cohort_score= CosineSimilarity()(cohort, cohort) #n,n 
for a in range(len(e)):
  for b in range(len(e)): 
    dsx.append({
        "enroll":e[a],
        "cohort":e[b],
        "score":enroll_cohort_score[a][b].item()
    })
enroll_cohort_score= pd.DataFrame(dsx)
preds=[]
cnt=0
# test=pd.read_csv("/content/dataroot/split_ds/imsv-public-test/imsv-public-test/public_test_cohart.csv")
from tqdm import tqdm
for index,row in tqdm(speaker_id.iterrows(),total=len(speaker_id)):
  test_key =os.path.join(f"{PATH_PRIVATE_ROOT}/",row.utterance_id) 
  key_emb = test_feat[test_key] # n,dim 
  c1= str(row.spkears[0]) 
  c2=str(row.spkears[1]) 
  c3=str(row.spkears[2])
  c4=str(row.spkears[3]) 
  c5=str(row.spkears[4]) 
  cohort = torch.cat([speaker_feat[c1], speaker_feat[c2], speaker_feat[c3],speaker_feat[c4],speaker_feat[c5]],0) #m,dim 
  e=[c1,] *len(speaker_feat[c1]) + [c2,] *len(speaker_feat[c2]) + [c3,] *len(speaker_feat[c3]) + [c4,] *len(speaker_feat[c4]) + [c5,] *len(speaker_feat[c5]) 
  # enr = speaker_feat[c1]
  # print(enr.size(), key_emb.size()) 
  score=CosineSimilarity()(cohort, key_emb).mean(-1)

  input_score = pd.DataFrame({"enroll":e,"test":[row.rand_ID,] *int(score.size(0)),"score":score.cpu().numpy().tolist()})
  dsx=[]
  enroll_cohort_score= CosineSimilarity()(cohort, cohort) #n,n 
  for a in range(len(e)):
    for b in range(len(e)): 
      dsx.append({
          "enroll":e[a],
          "cohort":e[b],
          "score":enroll_cohort_score[a][b].item()
      })

  enroll_cohort_score= pd.DataFrame(dsx)

  test_cohort_score=CosineSimilarity()(cohort, key_emb).mean(-1)
  test_cohort_score= pd.DataFrame({"test":[row.rand_ID,] *int(test_cohort_score.size(0)),"cohort":e,"score":test_cohort_score.cpu().numpy().tolist()})
  args=Args() 
  
  # if index!=3:continue
  # print(e,)
  sc=asnorm(args, input_score, enroll_cohort_score, test_cohort_score)
  sc = pd.DataFrame(sc)
  sc.columns=["c","wav","score"]
  
  l=sc.groupby("c")['score'].mean().sort_values().index[-1]
  sc=sc.groupby("c")['score'].mean().to_dict()
  if (l== str(row.Utterance_ID.split("_")[0])):
    cnt=cnt+1
  
  for k in sc:
    preds.append({
        "test":row.rand_ID,
        "score":sc[k],
        "spkear":k,
        "gt":str(k) == str(row.Utterance_ID.split("_")[0])
    })
p=pd.DataFrame(preds)
test=pd.read_csv(PATH_PRIVATE_PANDAS_SUB_SAMPLE)
scc=[]
dict_s={}
for index,row in p.iterrows():
  if row.test not in dict_s:
    dict_s[row.test]={}
  dict_s[row.test][row.spkear] = row.score
for index, row in test.iterrows():
  scc.append(
      dict_s[row.utterance_id][str(row.speaker_id)]
  )
test['score']=scc
# normalizer score to [0,1] 
min_score=test.groupby("utterance_id")['score'].apply(min).reset_index(name="min_cross")
max_score=test.groupby("utterance_id")['score'].apply(max).reset_index(name='max_cross')
test=pd.merge(test,max_score, on='utterance_id') 
test=pd.merge(test,min_score, on='utterance_id') 
test['score']  = (test['score'] - test['min_cross']) / (test['max_cross']-test['min_cross'])
test.to_csv("results.csv",index=False)






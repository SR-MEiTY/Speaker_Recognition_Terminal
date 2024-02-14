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
#from models.tdnn import ECAPA_TDNN_SMALL
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
a=torch.load("/home/iiitdwd/cocosda_wavlm/exps/exp_21thMay_rawnet_step3/epoch=1-VEER=49.963-mindcf=1.000.ckpt")['state_dict']
b={}
for k in a:
  if "model" not in k:continue
  b[k.replace("model.","")] = a[k]
model  = MainModelRawnet(
    nOut=256,
    sinc_stride=10,
    encoder_type="ECA"
)
model.load_state_dict(b)
testset = test_dataset_loader(
    list(glob.glob("/home/iiitdwd/cocosda_wavlm/data/Enr_data-20230204T124515Z-001/Enr_data/*.wav")),
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
model=model.to(device="cuda")
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
#for key in speaker_feat.keys():
  #speaker_feat[key] = torch.cat(speaker_feat[key],0)
torch.save(speaker_feat, "data_21stMay/env_emb_raw11thJune.pth")
print("Done corhort and test embedding")

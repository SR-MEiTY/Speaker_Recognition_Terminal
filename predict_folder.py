from models.tdnn import ECAPA_TDNN_SMALL
config_path = None
model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=config_path)
state=model.state_dict()
state_dict = torch.load("/kaggle/working/lastx.ckpt")['state_dict']
for k in state:
    state[k] = state_dict["model." + k]
model.load_state_dict(state, strict=False)
del state
del state_dict
import numpy as np

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

test=pd.read_csv("/kaggle/working/public_test/test_pairs/france_public_pairs.csv",header=None,sep=" ")

testset = test_dataset_loader(
    list(glob.glob("/kaggle/working/public_test_t1/France_Test/*.wav")),
    "",
    500,
    num_eval=5   
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
d_c=[]
x_c=[]
from tqdm import tqdm
for idx, data in tqdm(enumerate(test_loader),total=len(test_loader)):
    inp1 = data[0][0].cuda() #5,num_audio
    d_c.append(inp1)
    x_c.append(data[1][0])
    if len(d_c) == 16 or idx == len(test_loader)-1:
        d_c=torch.stack(d_c,0)# bs,5,num_audi
        nxx=d_c.size(1)
        bs=d_c.size(0)
#         print(d_c.size())
#         break
        with torch.no_grad():
            ref_feat = model(d_c.reshape(bs * nxx,-1)).detach().cpu()
            ref_feat=ref_feat.reshape(bs,nxx,-1).mean(1) # bs,-1
        for i,d in enumerate(x_c):
            feats[d] = ref_feat[i].reshape(1,-1)
        d_c=[]
        x_c=[]
torch.save(feats,"/kaggle/working/feat_france.pth")

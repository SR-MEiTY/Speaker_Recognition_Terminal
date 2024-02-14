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
#from models.tdnn import ECAPA_TDNN_SMALL
#from models import tdnn
config_path = None
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy, math
from tqdm import tqdm
from commons.loss_inter_aam import AAMINTER
wavlm_extractor = torch.hub.load('s3prl/s3prl',"wavlm_large")
from models.tdnn import ECAPA_TDNN_SMALL
model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None,extractor=wavlm_extractor)
checkpoint="/home/iiitdwd/cocosda_wavlm/exps/exp1_wavlm2/epoch=15-VEER=5.100-mindcf=0.198.ckpt"
b={}
#a = torch.load(checkpoint, map_location=lambda storage, loc: storage)
a = torch.load(checkpoint,map_location=torch.device('cpu'))
a=a['state_dict']
for k in a:
    if ("model.feature_extract.model") in k:
        b[k.replace("model.feature_extract.model.","feature_extract.model.")] = a[k]
    elif ("model" in k) and ("model.feature_extract.model") not in k:
        b[k.replace("model.","")] = a[k]
    else:
        continue
model.load_state_dict(b)
model.to(device="cpu")
model.eval()
testset = test_dataset_loader(
    list(glob.glob("/home/iiitdwd/cocosda_wavlm/data/I-MSV-Private-test-20230204T115807Z-001/I-MSV-Private-test/data/*.wav")),
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
model=model.to(device="cpu")
model.eval()
dc=[]
dx=[]
## Extract features for every image
for idx, data in tqdm(enumerate(test_loader),total=len(test_loader)):
    #inp1 = data[0][0].cuda() #bs,10,num_audio
    inp1 = data[0][0]
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
      torch.save(feats, "data_21stOct/test_emb_wavlm.pth")
    # feats[data[1][0]] = ref_feat
torch.save(feats, "data_21stOct/test_emb_wavlm_private.pth")




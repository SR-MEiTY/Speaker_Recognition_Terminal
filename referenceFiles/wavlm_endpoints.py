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


#######model loading#######
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
#==========================================================

########load audio#######
def loadWAV(filename, max_frames, evalmode=True, num_eval=10):

    # Maximum audio length
    max_audio = max_frames * 160 + 240
    #print(filename)
    # Read wav file and convert to torch tensor
    #print(filename)
    audio, sample_rate = soundfile.read(filename)
    #print("done====")

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
#==================================================


#######embedding extraction########
def speaker_embeddings(wavfile):
    audio=loadWAV(wavfile,500, evalmode=True, num_eval=15)
    data=torch.FloatTensor(audio)
    dc=[]
    d_c=torch.stack(dc,0)# bs,5,num_audio
    nxx=d_c.size(1)
    bs=d_c.size(0)
    with torch.no_grad():
        feat = model(d_c.reshape(bs * nxx,-1)).detach().cpu()
        feat=feat.reshape(bs,nxx,-1)
        return feat
    
#############scoring############
def scoring(enrol_embd,auth_embd):
    score=CosineSimilarity()(enrol_embd, auth_embd)
    return score
    

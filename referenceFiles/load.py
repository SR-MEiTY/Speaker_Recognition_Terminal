import torch
wavlm_extractor = torch.hub.load('s3prl/s3prl',"wavlm_large")
from models.tdnn import ECAPA_TDNN_SMALL
model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None,extractor=wavlm_extractor)
checkpoint="/home/iiitdwd/cocosda_wavlm/exps/exp1_wavlm2/epoch=15-VEER=5.100-mindcf=0.198.ckpt"
b={}
a = torch.load(checkpoint, map_location=lambda storage, loc: storage)
a=a['state_dict']
for k in a:
    if ("model.feature_extract.model") in k:
        b[k.replace("model.feature_extract.model.","feature_extract.model.")] = a[k]
    elif ("model" in k) and ("model.feature_extract.model") not in k:
        b[k.replace("model.","")] = a[k]
    else:
        continue
model.load_state_dict(b)

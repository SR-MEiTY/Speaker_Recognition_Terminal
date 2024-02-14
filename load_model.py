from models.tdnn import ECAPA_TDNN_SMALL
import torch
from pydub import AudioSegment
import numpy as np

cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

wavlm_extractor = torch.hub.load('s3prl/s3prl',"wavlm_large")   
model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
checkpoint="epoch=15-VEER=5.100-mindcf=0.198.ckpt"
b={}
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


def feature_extraction(full_file_name):
    audiodata = AudioSegment.from_file(full_file_name)
    audiodata = audiodata.set_frame_rate(16000)
    audio = audiodata.get_array_of_samples()

    sample_rate = audiodata.frame_rate

    sr=8000
    wavfilepath = 'enrolment.wav'

    max_frames = 500
    evalmode=True
    num_eval=10
    
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    audiosize = len(audio)

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = np.pad(audio, (0, shortage), 'wrap')
        audiosize   = len(audio) #.shape[0] newly added

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
    aud_feat = np.stack(feats,axis=0).astype(float)
    
    ##### Extract the Embeddings ######
    data=torch.FloatTensor(aud_feat)
    dc=[data]
    d_c=torch.stack(dc,0)# bs,5,num_audio
    nxx=d_c.size(1)
    bs=d_c.size(0)
    with torch.no_grad():
        feat = model(d_c.reshape(bs * nxx,-1)).detach().cpu()
        feat=feat.reshape(bs,nxx,-1)
    return feat


def scoring(speaker_feat, test_feat):
    score=cos(torch.mean(speaker_feat, dim=1), torch.mean(test_feat, dim=1))
    print(score)
    th = 0.5
    op=verify(score,th)
    if op == 1:
        output = "Speaker Recognized"
        return output;
    else:
        output = "Speaker not Recognized"
        return output;


def enrolment(id, audio_path):
    feat = feature_extraction(audio_path)
    model_save = 'enrolment/'+str(id)+".pth"
    torch.save(feat, model_save)
    print("Speaker Enrolment is Successful.")

def test(id, audio_path):
    feat = feature_extraction(audio_path)
    model_save = 'enrolment/'+str(id)+".pth"
    enrol_feat = torch.load(model_save)

    score = scoring(feat, enrol_feat)
    return score

def verify(score,th):
    if score>=th:
        op=1
    else:
        op=0
    return op
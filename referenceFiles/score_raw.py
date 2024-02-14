from pytorch_metric_learning.distances import BatchedDistance, CosineSimilarity
import os
import torch
import argparse
import pandas as pd

from tqdm import tqdm
import sys
import logging
import argparse
import traceback
import pandas as pd
def asnorm(input_score, enroll_cohort_score,test_cohort_score):
    """ Adaptive Symmetrical Normalization.
    Reference: Cumani, S., Batzu, P. D., Colibro, D., Vair, C., Laface, P., & Vasilakakis, V. (2011). Comparison of 
               speaker recognition approaches for real applications. Paper presented at the Twelfth Annual Conference 
               of the International Speech Communication Association.
               Cai, Danwei, et al. “The DKU-SMIIP System for NIST 2018 Speaker Recognition Evaluation.” Interspeech 2019, 
               2019, pp. 4370–4374.
    Recommend: Matejka, P., Novotný, O., Plchot, O., Burget, L., Sánchez, M. D., & Cernocký, J. (2017). Analysis of 
               Score Normalization in Multilingual Speaker Recognition. Paper presented at the Interspeech.
    """
    enroll_test_names = ["enroll", "test", "score"]
    enroll_cohort_names = ["enroll", "cohort", "score"]
    test_cohort_names = ["test", "cohort", "score"]
    # input_score = load_score(args.input_score, enroll_test_names)
    # enroll_cohort_score = load_score(args.enroll_cohort_score, enroll_cohort_names)
    # test_cohort_score = load_score(args.test_cohort_score, test_cohort_names)

    output_score = []

    # logger.info("Use Adaptive Symmetrical Normalization (AS-Norm) to normalize scores ...")

    # Note that, .sort_values function will return NoneType with inplace=True and .head function will return a DataFrame object.
    # The order sort->groupby is equal to groupby->sort, so there is no problem about independence of trials.
    enroll_cohort_score.sort_values(by="score", ascending=False, inplace=True)
    test_cohort_score.sort_values(by="score", ascending=False, inplace=True)
    xxx=False
    if xxx is True:
        # logger.info("Select top n scores by cross method.")
        # The SQL grammar is used to implement the cross selection based on pandas.
        # Let A is enroll_test table, B is enroll_cohort table and C is test_cohort table.
        # To get a test_group (select "test:cohort" pairs) where the cohort utterances' scores is selected by enroll_top_n,
        # we should get the D table by concatenating AxC with "enroll" key firstly and then
        # we could get the target E table by concatenating BxD wiht "test"&"cohort" key.
        # Finally, the E table should be grouped by "enroll"&"test" key to make sure the group key is unique.
        enroll_top_n = enroll_cohort_score.groupby("enroll").head(args.top_n)[["enroll", "cohort"]]
        test_group = pd.merge(
            pd.merge(input_score[["enroll", "test"]], enroll_top_n, on="enroll"), 
            test_cohort_score, on=["test", "cohort"]).groupby(["enroll", "test"])

        test_top_n = test_cohort_score.groupby("test").head(args.top_n)[["test", "cohort"]]
        enroll_group = pd.merge(pd.merge(input_score[["enroll", "test"]], test_top_n, on="test"), 
                                enroll_cohort_score, on=["enroll", "cohort"]).groupby(["enroll", "test"])
    else:
        enroll_group = enroll_cohort_score.groupby("enroll").head(3).groupby("enroll")
        test_group = test_cohort_score.groupby("test").head(3).groupby("test")

    enroll_mean = enroll_group["score"].mean()
    enroll_std = enroll_group["score"].std()
    test_mean = test_group["score"].mean()
    test_std = test_group["score"].std()

    for _, row in input_score.iterrows():
        enroll_key, test_key, score = row
        if xxx is True:
            normed_score = 0.5 * ((score - enroll_mean[enroll_key, test_key]) / enroll_std[enroll_key, test_key] + \
                                 (score - test_mean[enroll_key, test_key]) / test_std[enroll_key, test_key])
        else:
            normed_score = 0.5 * ((score - enroll_mean[enroll_key]) / enroll_std[enroll_key] + \
                                (score - test_mean[test_key]) / test_std[test_key])
        output_score.append([enroll_key, test_key, normed_score])

    return output_score
args=argparse.ArgumentParser()
speaker_feat=torch.load("/home/iiitdwd/cocosda_wavlm/data_21stMay/enrol_emb_raw_11thJune.pth")
test_feat=torch.load("/home/iiitdwd/cocosda_wavlm/data_21stOct/test_emb_raw_private.pth")
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
        "score":enroll_cohort_score[a][b].item()})
enroll_cohort_score= pd.DataFrame(dsx)
preds=[]
cnt=0
speaker_id=pd.read_csv("/home/iiitdwd/cocosda_wavlm/data/I-MSV-Private-test-20230204T115807Z-001/I-MSV-Private-test/private_test_cohart.csv")
from tqdm import tqdm
for index,row in tqdm(speaker_id.iterrows(),total=len(speaker_id)):
    test_key =os.path.join("/home/iiitdwd/cocosda_wavlm/data/I-MSV-Private-test-20230204T115807Z-001/I-MSV-Private-test/data",row.utterance_id) 
    print(test_key)
    print(test_key)
    key_emb = test_feat[test_key] # n,dim 
    c1= str(row.c1) 
    c2=str(row.c2) 
    c3=str(row.c3)
    c4=str(row.c4) 
    c5=str(row.c5) 
    cohort = torch.cat([speaker_feat[c1], speaker_feat[c2], speaker_feat[c3],speaker_feat[c4],speaker_feat[c5]],0) #m,dim 
    e=[c1,] *len(speaker_feat[c1]) + [c2,] *len(speaker_feat[c2]) + [c3,] *len(speaker_feat[c3]) + [c4,] *len(speaker_feat[c4]) + [c5,] *len(speaker_feat[c5]) 
    # enr = speaker_feat[c1]
    # print(enr.size(), key_emb.size()) 
    score=CosineSimilarity()(cohort, key_emb).mean(-1)
    input_score = pd.DataFrame({"enroll":e,"test":[row.utterance_id,] *int(score.size(0)),"score":score.cpu().numpy().tolist()})
    dsx=[]
    enroll_cohort_score= CosineSimilarity()(cohort, cohort) #n,n 
    for a in range(len(e)):
        for b in range(len(e)):
            dsx.append({
          "enroll":e[a],
          "cohort":e[b],
          "score":enroll_cohort_score[a][b].item()})
    enroll_cohort_score= pd.DataFrame(dsx)
    test_cohort_score=CosineSimilarity()(cohort, key_emb).mean(-1)
    test_cohort_score= pd.DataFrame({"test":[row.utterance_id,] *int(test_cohort_score.size(0)),"cohort":e,"score":test_cohort_score.cpu().numpy().tolist()})

    # if index!=3:continue
    # print(e,)
    #sc=asnorm( input_score, enroll_cohort_score, test_cohort_score)
    sc=input_score
    sc = pd.DataFrame(sc)
    sc.columns=["c","wav","score"]
    l=sc.groupby("c")['score'].mean().sort_values().index[-1]
    sc=sc.groupby("c")['score'].mean().to_dict()
    if (l== str(row.Utterance_ID.split("_")[0])):
        cnt=cnt+1
    for k in sc:
        preds.append({
        "test":row.utterance_id,
        "score":sc[k],
        "speaker_id":k,
        "gt":str(k) == str(row.Utterance_ID.split("_")[0])})
p=pd.DataFrame(preds)
p.to_csv("preds_asnorm_raw_private_nonorm.csv")

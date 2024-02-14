import scipy
import numpy as np
import math
import os
import sys
from plda_base import PldaStats,PldaEstimation
import logging

def compute_plda(spk2vectors, plda_out="./plda"):
    """spk2vectors={
        "spk1":[emb11, emb12,...],
        "spk2::[emb21,emb22,emb23,...]
    }
    emb11: shape[dim,1] or shape[dim,]
    """

    spks = list(spk2vectors.keys())
    dim = spk2vectors[spks[0]][0].shape[0]
    plda_stats=PldaStats(dim)
    for key in spk2vectors.keys():
        vectors = np.array(spk2vectors[key], dtype=float)
        weight = 1.0
        plda_stats.add_samples(weight,vectors)
    print("Estimate the parameters of PLDA by EM algorithm...")
    plda_stats.sort()
    plda_estimator=PldaEstimation(plda_stats)
    plda_estimator.estimate()
    print('Save the parameters for the PLDA adaptation...')
    plda_estimator.plda_write(plda_out+'.ori')
    plda_trans = plda_estimator.get_output()
    logger.info('Save the parameters for scoring directly, which is the same with the plda in kaldi...')
    plda_trans.plda_trans_write(plda_out+".txt")

    print("Training done PLDA")





def PLDAScoring(enroll_xvector,test_xvector,Gamma,Lambda,c,k):

    score = np.matmul(np.matmul(enroll_xvector.T,Lambda),test_xvector) + np.matmul(np.matmul(test_xvector.T,Lambda),enroll_xvector) \
        + np.matmul(np.matmul(enroll_xvector.T,Gamma),enroll_xvector) + np.matmul(np.matmul(test_xvector.T,Gamma),test_xvector) \
        + np.matmul((enroll_xvector + test_xvector).T,c) + k
    
    return score[0][0]

def CalculateVar(between_var,within_var,mean):

    total_var_inv = np.linalg.inv(between_var + within_var)
    wc_add_2ac_inv = np.linalg.inv(within_var + 2* between_var)
    wc_inv = np.linalg.inv(within_var)

    # Gamma
    Gamma = (-1/4)*(wc_add_2ac_inv+wc_inv)+(1/2)*total_var_inv

    #Lambda
    Lambda = (-1/4)*(wc_add_2ac_inv-wc_inv)

    #c
    c = np.matmul((wc_add_2ac_inv-total_var_inv),mean)
    
    # Since k is a constant for the addition of all scores and does not affect the eer value, it is not counted as a scoring term
    # k = logdet_tot -(1/2) *(logdet_w_two_b+logdet_w) + np.matmul(np.matmul(mean.T,np.linalg.inv(total_var)-np.linalg.inv(within_var + 2* between_var)),mean)
    k = 0

    return Gamma,Lambda,c,k


def predict(plda, enroll_xvector,test_xvector,trials):
    '''plda: str path to plda model
       *_xvector:{
            "utt":np.ndarray(dim)
       }
       trials:[
            (utt_enroll1, utt_test1, *),
            (utt_enroll2, utt_test2, *),
       ]
    '''
    with open(plda,"rb") as f:
        mean = np.load(f)
        dim = mean.shape[0]
        within_var = np.load(f)
        within_var =within_var.reshape(dim, dim)
        between_var = np.load(f).reshape(dim, dim)
    within_var = within_var + 5e-5*np.eye(within_var.shape[0])
    Gamma,Lambda,c,k = CalculateVar(between_var,within_var,mean)
    scores=[]
    for trial in trials:
        enroll,test= trial[0], trial[1]
        core = PLDAScoring(enroll_xvector[enroll].reshape(-1,1),test_xvector[test].reshape(-1,1),\
                                                                Gamma,Lambda,c,k)
        scores.append(core)
    return scores



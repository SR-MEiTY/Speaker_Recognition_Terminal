#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:24:13 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, IIT Dharwad
"""

from flask import Flask, render_template, request, make_response, flash, redirect
import csv
import os
#Packages required for training and verification of the audio files
import numpy as np
import librosa as lb
os.environ["FLASK_RUN_PORT"] = '443'
from sklearn.mixture import GaussianMixture as GMM
import pickle
import warnings
import io, zipfile, time
from scipy.io import wavfile
import librosa
import requests
import json
warnings.filterwarnings('ignore')



#Directories to store audio data of speaker for training and testing 
TRAIN_FOLDER = 'static/train_data/'
TEST_FOLDER = 'static/test_data/'
FILEPATH = 'static/data/key_files_edited.zip'

#Flask Server Instance
app = Flask(__name__)

def convert_to_pcm(file_name):
    audio, sr = librosa.load(file_name, sr=None)
    wavfile.write(file_name, sr, audio)

#Default route to load home page of the web app
@app.route('/', methods = ['POST', 'GET'])
def root():
	return render_template('index.html');

@app.route('/index.html', methods = ['POST', 'GET'])
def index():
	return render_template('index.html');

@app.route('/about.html', methods = ['POST', 'GET'])
def about():
	return render_template('about.html');

@app.route('/Available_Toolkits.html', methods = ['POST', 'GET'])
def Available_Toolkits():
	return render_template('Available_Toolkits.html');

@app.route('/feedback.html', methods = ['POST', 'GET'])
def feedback():
	return render_template('feedback.html');

@app.route('/hands_on.html', methods = ['POST', 'GET'])
def hands_on():
	return render_template('hands_on.html');

@app.route('/I-MSV.html', methods = ['POST', 'GET'])
def I_MSV():
	return render_template('I-MSV.html');

@app.route('/references.html', methods = ['POST', 'GET'])
def references():
	return render_template('references.html');

@app.route('/resources.html', methods = ['POST', 'GET'])
def resources():
	return render_template('resources.html');

@app.route('/data_key', methods = ['POST', 'GET'])
def data_key_download():
# 	return render_template('resources.html');
    fileobj = io.BytesIO()
    with zipfile.ZipFile(fileobj, 'w') as zip_file:
        zip_info = zipfile.ZipInfo(FILEPATH)
        zip_info.date_time = time.localtime(time.time())[:6]
        zip_info.compress_type = zipfile.ZIP_DEFLATED
        with open(FILEPATH, 'rb') as fd:
            zip_file.writestr(zip_info, fd.read())
    fileobj.seek(0)

    response = make_response(fileobj.read())
    response.headers.set('Content-Type', 'zip')
    response.headers.set('Content-Disposition', 'attachment', filename='%s.zip' % os.path.basename(FILEPATH))
    return response

@app.route('/self_evaluation.html', methods = ['POST', 'GET'])
def self_evaluation():
	return render_template('self_evaluation.html');

@app.route('/SRS.html', methods = ['POST', 'GET'])
def SRS():
	return render_template('SRS.html');

@app.route('/demo.html', methods = ['POST', 'GET'])
def demo():
	return render_template('demo.html');

@app.route('/demo_wavlm.html', methods = ['POST', 'GET'])
def demo_wavlm():
	return render_template('demo_wavlm.html');

#This route load the registration page for the user
@app.route('/registration.html', methods = ['POST', 'GET'])
def registration():
	return render_template('registration.html');

#This route loads the verification page for the user
@app.route('/verification.html', methods = ['POST', 'GET'])
def testing():
	return render_template('verification.html');

#This route load the registration page for the user
@app.route('/registration_wavlm.html', methods = ['POST', 'GET'])
def registration_wavlm():
	return render_template('registration_wavlm2.html');

#This route loads the verification page for the user
@app.route('/verification_wavlm.html', methods = ['POST', 'GET'])
def testing_wavlm():
	return render_template('verification_wavlm2.html');

@app.route('/VOA.html', methods = ['POST', 'GET'])
def VOA():
	return render_template('VOA.html');


@app.route('/VOA_registration.html', methods = ['POST', 'GET'])
def VOA_registration():
	return render_template('VOA_registration.html');


@app.route('/VOA_verification.html', methods = ['POST', 'GET'])
def VOA_verification():
	return render_template('VOA_verification.html');


@app.route('/VOA_submitfiles', methods=['POST'])
def submit_audio_files():
	fname = request.form['fname']
	lname = request.form['lname']
	age = request.form['age']
	gender = request.form['gender']
	
	audio_files = request.files.getlist('files[]')
	print("__________________________Local_________________________")

	backend_url = 'http://10.250.1.59:8082/backend_enrollment'  

	audio_files = [('file', file) for file in audio_files]
	print(audio_files)

	data = {'fname':fname, 'lname':lname,'gender': gender, 'age': age,  }
	row = [fname, lname, gender, age]
	csvfile = open('static/data/VOA_speaker_info.csv', 'w', newline='')
	writer = csv.writer(csvfile)
	writer.writerow(row)
	csvfile.close()

	try:
		response = requests.post(backend_url, files=audio_files, data=data)
		
		if response.status_code == 200:
			result = response.text
		else:
			result = "Error communicating with the backend server"
	except Exception as e:
		result = str(e)
	print(result)
	return result

@app.route('/VOA_submitverify', methods=['GET','POST'])
def VOA_verify_submit():
    if request.method == 'POST':
        speaker_label = request.form.get('speaker_label')
        otp = request.form.get('otp')
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print("__________________________Local_________________________")
        # Send the audio file to the backend server
        backend_url = 'http://10.250.1.59:8082/backend_verification'  # Replace with the actual backend endpoint
        files = {'file': file}
        data = {'speaker_label': speaker_label, 'otp': otp}
        try:
            response = requests.post(backend_url, files=files, data=data)   
            if response.status_code == 200:
                result = response.text
            else:
                result = "Error communicating with the backend server"
        except Exception as e:
            result = str(e)
        return result


#This route performs the following functionalities:
#Fetches speaker details from the HTML form and store it in the csv file
#Creates the speaker ID
#Creates the training and testing directories for the speaker with created speaker IS
#Stores the audio file from the client broswer in current speaker's training directory
#Extracts the audio features and generates the .gmm file for further usage
@app.route('/uploadTAudio', methods = ['GET', 'POST'])
def uploadTAudio():
	if request.method == 'POST':
		file = request.files['audioChunk'];
		firstName = request.form['firstName'];
		lastName = request.form['lastName'];
		gender = request.form['gender'];
		age = request.form['age'];
		
		speakerID = firstName[0].upper()+lastName[0].upper()+gender+age;
		
		row = [firstName, lastName, gender, age, speakerID]

		isTraindir = os.path.isdir(TRAIN_FOLDER + speakerID)
		isTestdir = os.path.isdir(TEST_FOLDER + speakerID)
		print(isTraindir)
		print(isTestdir)
		if isTraindir == False and isTestdir == False: 
			os.mkdir(TRAIN_FOLDER + speakerID);
			os.mkdir(TEST_FOLDER + speakerID);
			print('Training directory for ' + speakerID +' created successfully.');
			print('Verification directory for ' +speakerID +' created successfully.');
		else:
			print('Directory already exists for the speaker!')
			print('Skipping the directory creation for Training Data...')
			print('Skipping the directory creation for Testing Data...')
		csvfile = open('static/speakerinfo.csv', 'a', newline='');
		writer = csv.writer(csvfile);
		writer.writerow(row);
		csvfile.close();
		file_name = speakerID + ".wav"
		full_file_name = os.path.join(TRAIN_FOLDER+speakerID, file_name)
		file.save(full_file_name)

		path2str  = "static/train_data/"  
		sid=speakerID
		sr=8000

		wavfilepath=path2str + sid + '/' + sid + '.wav'
		y, sr = lb.load(wavfilepath, sr=sr)
		features = extract_features(y,sr)
		gmm = GMM(n_components = 16, max_iter=50, n_init = 3)
		gmm.fit(features)
		model_save = path2str + sid + '/' + sid + ".gmm"
		pickle.dump(gmm,open(model_save,'wb'))
		if isTraindir and isTestdir:
			return_string = sid + ' already exists. Overwriting the existing file, if any...'
		else:
			return_string = 'Please note down your Speaker ID: ' + sid

		return return_string



#This route performs the following functionalities:
#Fetches the speaker ID from the web browser and checks whether it is in the testing direcory or not
#Fetches the audio from web browser and stores it in the current speaker's testing directory
#Extracts the features from the audio
#Loads the .gmm file from the current user's training directory and calculates the score
#based on the score it returns whether speaker is recognized or not.
@app.route('/uploadVAudio', methods = ['POST'])
def uploadVAudio():
	if request.method == 'POST':
		file = request.files['audioChunk'];
		sid = request.form['sid'].upper();

		csv_file = open('static/speakerinfo.csv', "r");
		reader = csv.reader(csv_file)
		isFound = False
		for row in reader:
			if sid == row[4]:
				isFound = True
				break
			else:
				continue
			print(isFound)
		if isFound:
			file_name = sid + ".wav"
			full_file_name = os.path.join(TEST_FOLDER+sid, file_name)
			file.save(full_file_name)

			path2str  = "static/test_data/"  
			#id=sid
			sr=8000
			#%%
			wavfilepath=path2str + sid + '/' + sid + '.wav'
			y,sr = lb.load(wavfilepath, sr=sr)
			features = extract_features(y,sr)
			#print(f'features={np.shape(features)}')

			speaker_model_path = 'static/train_data/' + sid + '/' + sid + '.gmm'
			speaker_model = pickle.load(open(speaker_model_path,'rb'))
			speaker_score = speaker_model.score(features)

			ubm_model_path = 'static/train_data/ubm.pkl'
			ubm_model = pickle.load(open(ubm_model_path,'rb'))['model']
			ubm_score = ubm_model.score(features)

			score = speaker_score-ubm_score

			th=7
			print(f'Speaker score={score} {speaker_score} {ubm_score} threshold={th}')
			op = verify(score,th)
			#print(f'verification status={op} (1=Recognized; 0=Not Recognized)')
			if op == 1:
				#output = "Speaker Recognized"
				return "1";
			else:
				#output = "Speaker not Recognized"
				return "0";
		else:
			return "-1"

#This function is used to extract the features from the audio
def extract_features(y,sr):
	mfcc = lb.feature.mfcc(y=y, sr=sr,n_mfcc=14)
	mfcc_delta = lb.feature.delta(mfcc)
	mfcc_delta2 = lb.feature.delta(mfcc, order=2)

	mfcc = mfcc[1:]
	mfcc_delta = mfcc_delta[1:]
	mfcc_delta2 = mfcc_delta2[1:]
	combined = np.hstack((mfcc.T,mfcc_delta.T, mfcc_delta2.T)) 
	return combined

#This function is compares the score with threshold and return 0 or 1 based on the comparison
def verify(score,th):
    if score>=th:
        op=1
    else:
        op=0
    return op

### LIBRARIES REQUIRED FOR POPCORN SPEAKER RECOGNITION #####################################################################################
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
#import time, pdb, numpy, math
import time, pdb, math
from tqdm import tqdm
from commons.loss_inter_aam import AAMINTER
import numpy as np
import soundfile
from pydub import AudioSegment
###########################################################################################################################################

### LOADING THE MODELS ####################################################################################################################
#from models.tdnn import ECAPA_TDNN_SMALL
wavlm_extractor = torch.hub.load('s3prl/s3prl',"wavlm_large")
from models.tdnn import ECAPA_TDNN_SMALL
#model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None, extractor=wavlm_extractor)
model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
checkpoint="epoch=15-VEER=5.100-mindcf=0.198.ckpt"
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
############################################################################################################################################

#Directories to store audio data of speaker for training and testing #####
TRAIN_FOLDER = 'static/train_data/'
TEST_FOLDER = 'static/test_data/'
##########################################################################

@ app.route('/uploadTTAudio', methods = ['POST'])
def uploadTTAudio():
	if request.method == 'POST':
		file = request.files['audioChunk'];
		firstName = request.form['firstName'];
		lastName = request.form['lastName'];
		gender = request.form['gender'];
		age = request.form['age'];
		
		speakerID = firstName[0].upper()+lastName[0].upper()+gender+age;
		
		row = [firstName, lastName, gender, age, speakerID]

		isTraindir = os.path.isdir(TRAIN_FOLDER + speakerID)
		isTestdir = os.path.isdir(TEST_FOLDER + speakerID)
		print(isTraindir)
		print(isTestdir)
		if isTraindir == False and isTestdir == False: 
			os.mkdir(TRAIN_FOLDER + speakerID);
			os.mkdir(TEST_FOLDER + speakerID);
			print('Training directory for ' + speakerID +' created successfully.');
			print('Verification directory for ' +speakerID +' created successfully.');
		else:
			print('Directory already exists for the speaker!')
			print('Skipping the directory creation for Training Data...')
			print('Skipping the directory creation for Testing Data...')
		csvfile = open('static/speakerinfo.csv', 'a', newline='');
		writer = csv.writer(csvfile);
		writer.writerow(row);
		csvfile.close();
		file_name = speakerID + ".wav"
		full_file_name = os.path.join(TRAIN_FOLDER+speakerID, file_name)
		file.save(full_file_name)
		# audio_binary_data = file.read()

		# With Somashekhar
		
		audiodata = AudioSegment.from_file(full_file_name)
		audiodata = audiodata.set_frame_rate(16000)
		audio = audiodata.get_array_of_samples()

		sample_rate = audiodata.frame_rate

		path2str  = "static/train_data/"  
		id=speakerID
		sr=8000
		wavfilepath=path2str+id+ '/'+id+'.wav'

		max_frames = 500
		evalmode=True
		num_eval=10
		
		# Maximum audio length
		max_audio = max_frames * 160 + 240
		#print(filename)
		# Read wav file and convert to torch tensor
		#print(filename)
		# audio, sample_rate = soundfile.read(wavfilepath)
		#print("done====")

		# audiosize = audio.shape[0]
		#Newly Added
		audiosize = len(audio)

		if audiosize <= max_audio:
		    shortage    = max_audio - audiosize + 1 
		    audio       = np.pad(audio, (0, shortage), 'wrap')
		    audiosize   = len(audio) #.shape[0]	newly added

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
			#return feat
			model_save = path2str+id+'/'+id+".pth"
			torch.save(feat, model_save)
	return id

#########################################################################################################

#########################################################################################################
#This route performs the following functionalities:
#Fetches the speaker ID from the web browser and checks whether it is in the testing direcory or not
#Fetches the audio from web browser and stores it in the current speaker's testing directory
#Extracts the features from the audio
#Loads the embedding file from the current user's training directory and calculates the score
#based on the score it returns whether speaker is recognized or not.
@ app.route('/uploadVVAudio', methods = ['POST'])
def uploadVVAudio():
	if request.method == 'POST':
		file = request.files['audioChunk'];
		sid = request.form['sid'].upper();

		csv_file = open('static/speakerinfo.csv', "r");
		reader = csv.reader(csv_file)
		isFound = False
		for row in reader:
			if sid == row[4]:
				isFound = True
				break
			else:
				continue
			print(isFound)
		if isFound:
			file_name = sid + ".wav"
			full_file_name = os.path.join(TEST_FOLDER+sid, file_name)
			file.save(full_file_name)

			path2str  = "static/test_data/"  
			id=sid
			sr=8000
			wavfilepath=path2str+id+ '/'+id+'.wav'

		max_frames = 500
		evalmode=True
		num_eval=10
		
		# Maximum audio length
		max_audio = max_frames * 160 + 240
		#print(filename)
		# Read wav file and convert to torch tensor
		#print(filename)
		#audio, sample_rate = soundfile.read(wavfilepath)
		#print("done====")

		audiodata = AudioSegment.from_file(full_file_name)
		audiodata = audiodata.set_frame_rate(16000)
		audio = audiodata.get_array_of_samples()
		sample_rate = audiodata.frame_rate
		print(sample_rate)

		#audiosize = audio.shape[0]
		audiosize   = len(audio)
		print(audiosize)
		print(max_audio)

		if audiosize <= max_audio:
		    shortage    = max_audio - audiosize + 1 
		    audio       = np.pad(audio, (0, shortage), 'wrap')
		    audiosize   = len(audio) #.shape[0]

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
			#return feat
			model_save = path2str+id+'/'+id+".pth"
			torch.save(feat, model_save)

			# SCORING FUNCTION
			path2enrol  = "static/train_data/"
			#enrol_embd = path2enrol+id+'/'+id+".pth"
			#auth_embd = path2str+id+'/'+id+".pth"

			print(path2enrol)
			print(path2str)

			speaker_feat=torch.load(path2enrol+id+'/'+id+".pth")
			test_feat=torch.load(path2str+id+'/'+id+".pth")
			print("Printing")
			print(speaker_feat.size())
			print(torch.mean(speaker_feat, dim=1).size())
			print(type(speaker_feat))
			score=CosineSimilarity()(torch.mean(speaker_feat, dim=1), torch.mean(test_feat, dim=1))
			print(score)
			th = 0.5
			op=verify(score,th)
			if op == 1:
				#output = "Speaker Recognized"
				return "1";
			else:
				#output = "Speaker not Recognized"
				return "0";

		# 	y,sr=lb.load(wavfilepath,sr=sr)
		# 	features  = extract_features(y,sr)

		# 	model_load='static/train_data/'+id;

		# 	model_load_path=model_load+'/'+id+'.gmm'
		# 	model=pickle.load(open(model_load_path,'rb'))

		# 	score=model.score(features)

		# 	th=-100
		# 	op=verify(score,th)
		# 	print(op)
		# 	if op == 1:
		# 		#output = "Speaker Recognized"
		# 		return "1";
		# 	else:
		# 		#output = "Speaker not Recognized"
		# 		return "0";
		# else:
		# 	return "-1"
#################################################################################################


@ app.route('/checkUser', methods = ['POST'])
def checkUser():
	if request.method == 'POST':
		data = request.get_json();
		speakerID = data['username'];

		print("Speaker ID:", speakerID)
		csv_file = open('static/speakerinfo.csv', "r");
		reader = csv.reader(csv_file)
		isFound = False
		for row in reader:
			if speakerID == row[4]:
				isFound = True
				break
			else:
				continue
			print(isFound)
		if isFound:
			return "1"
		else:
			return "-1"



if __name__ == '__main__':
    app.config['TRAIN_FOLDER'] = TRAIN_FOLDER
    app.config['TEST_FOLDER'] = TEST_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
    #app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
    # Local
    #context = ('certificate.pem', 'privateKey.pem')
    #app.run(host="127.0.0.1", debug=True, port=8888)
    
    wavlm_extractor = torch.hub.load('s3prl/s3prl',"wavlm_large")   
    from models.tdnn import ECAPA_TDNN_SMALL
    #model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None, extractor=wavlm_extractor)
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large', config_path=None)
    checkpoint="epoch=15-VEER=5.100-mindcf=0.198.ckpt"
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
    
    # IIT-Dh server
    context = ('flaskssl/certificate.crt', 'flaskssl/star_iitdh_key.key')
    app.run(host="0.0.0.0", debug=True, port=6060, ssl_context=context, )
    

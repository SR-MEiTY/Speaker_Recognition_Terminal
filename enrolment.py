from load_model import enrolment, test
import os

os.makedirs('enrolment', exist_ok=True)

# Enrolment ID and audio path for enrolment of speaker
enrol_id = 123
audio_path = "70_1_6526_F_20@0&1_21-12-2022_15-36-13_BT.wav"
enrolment(enrol_id, audio_path)
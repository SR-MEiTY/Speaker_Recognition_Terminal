from load_model import enrolment, test
import os

# Claim ID i.e. ID for verification and new test audio path
claim_id = 123
audio_path = "70_1_6526_F_20@0&1_21-12-2022_15-36-13_BT.wav"
res = test(claim_id, audio_path)
print(res)
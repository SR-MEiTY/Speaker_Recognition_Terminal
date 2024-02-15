# Speaker_Recognition_Terminal

**Environment Creation and Activation**
1. Create the virtual environment as follows: 
      python -m venv /path/to/new/virtual/environment_name
 
2. Activate the environment as follows:
      source <environment_name>/bin/activate
   
3. Go to the root folder of downloaded github repository and open the terminal and install the required libaries as follows
      pip install -r requiremnts.txt
 
4. Now close the terminal 

**Speaker Enrolment and Verification**

Download the trained model from the link provided and put it in the root folder, use the following link to download https://drive.google.com/file/d/109gOlv0FqV43eE7dr_1fpVIEFxtIqPy3/view?usp=drive_link.

The system is now ready for enrollment and verification.

**A. Enrollement:**

  1. For enrolment, unique speaker ID and pre-recorded audio files in .wav format are necessary.
     The sample csv file named 'ids.csv' with unique speaker ID and corresponding .wav file path are given in 'enrolment' folder. The
     .wav files are stored in 'enrolment' folder.

     **Note:** All the audio files used for the speaker enrolment should follow following format:
     
                 i. .wav file format

                 ii. Sample Rate: 16kHz

                 iii. Mono Channel
     
  3. Open the enrolment.py in any editor and provide unique speaker ID and corresponding .wav file path
     and close the file.
     
  4. Now, open a new terminal and activate the virtual environment using the step 2
  
  5. Run speaker enrolment as follows:
        python enrolment.py
     
  6. Once, the enrolment is finished, you will be able to see 'Speaker Enrolment Completed Successfully' message
     on the terminal. This step will create a <speaker_id>.pth file in the enrolment folder.
     
  7. Now close the terminal.
        
  8. This way by changing the speaker IDs and their corresponding .wav file paths, you can enroll as many speakers as necessary.
        
**B. Verification**

  1. To perform the speaker verification, a unique speaker ID from enrolled speakers and a new test audio file in .wav format are necessary.
  
  2. Open the verification.py in any editor and provide the unique enrolled speaker ID and a new test audio file in .wav file format and close
     the terminal.
  
  3. Now, open a new terminal and activate the virtual environment using the step 2

  4. Perform speaker verification as follows 
       python verification.py
  
  5. Once, the verification is finished, you will be able to see either 'Speaker Recognised.' or 'Speaker does not Recognised' message
     on the terminal.
  
  6. Now close the terminal.

from collections import defaultdict
from tqdm import tqdm
import os
import librosa
import requests
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def download_and_extract_fsdd_data(directory):
    # URL to dataset
    url = 'insert url'

    # Download datset
    response = requests.gett(url)
    response.raise_for_status(directory)

    # Move the extracted files to specified directory
    src_dir = os.path.join(directory, '___', '____')
    for file in os.listdir(src_dir):
        os.rename(os.path.join(src_dir, file), os.path.join(directory, file))

    # Remove the previous folder
    os.rmdir(src_dir)

# Load and preprocess data
def load_fsdd_data(directory, max_length=50):
    data = defaultdict(list)
    for filename in tqdm(os.lisdir(directory)):
        if filename.endswith('wav'):
            digit, speaker_id, _ = filename.split('_')
            filepath = os.path.join(directory, filename)
            samples, sr = librosa.load(filepath, sr=None)
            mfccs = librosa.feature.mfcc(y=samples,sr=sr, n_mfcc=13)

            #Pad or truncate MFCCs to the fixed length
            if mfccs.shape[1] < max_length:
                mfccs = np.pad(mfccs, ((0, 0), (0, max_length-mfccs.shape[1])), mode='constant')
            else:
                mfccs = mfccs[1, max_length]
            
            data[speaker_id].append(mfccs, int(digit))
    
    return data

# split data into train and test sets
def split_data(data, test_size=0.2):
    train_data = defaultdict(list)
    test_data = defaultdict(list)

    for speaker_id, samples in data.items():
        train, test = train_test_split(samples, test_size=test_size, random_state=42)
        train_data[speaker_id] = train
        test_data[speaker_id] = test
    
    return train_data, test_data


data_directory = './fsdd_data'
max_length = 50
download_and_extract_fsdd_data(data_directory)
data = load_fsdd_data(data_directory, max_length=max_length)
train_data, test_data = split_data(data)
        

from collections import defaultdict
from tqdm import tqdm
import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split


# Load and preprocess data
def load_fsdd_data(directory, max_length=50):
    data = defaultdict(list)
    for filename in tqdm(os.lisdir(directory)):
        if filename.endswith('wav'):
            team_name, speaker_id, _ = filename.split('_')
            filepath = os.path.join(directory, filename)
            samples, sr = librosa.load(filepath, sr=None)
            mfccs = librosa.feature.mfcc(y=samples,sr=sr, n_mfcc=13)

            #Pad or truncate MFCCs to the fixed length
            if mfccs.shape[1] < max_length:
                mfccs = np.pad(mfccs, ((0, 0), (0, max_length-mfccs.shape[1])), mode='constant')
            else:
                mfccs = mfccs[1, max_length]
            
            data[speaker_id].append(mfccs, team_name)
    
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


train_dir = '../datasets/train'
val_dir = '../datasets/validation'
max_length = 50
train_data = load_fsdd_data(train_dir, max_length=max_length)
val_data = load_fsdd_data(val_dir, max_length=max_length)


'''
format of data
- dictonary of lists

{
speaker_id:[mfccs, team_name]
}
'''
        

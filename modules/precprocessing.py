from collections import defaultdict
from tqdm import tqdm
import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split

from split_wav import SplitWavAudioMubin
from audio_augment import stretch, shift, pitch
from denoiser import denoise

# Split 15s audio into 3s segments with overlap of 1s, ignore the first 1s when there is no sound
def split_audio_data(directory, audio_length=3, audio_overlap=1, start_sec=1):
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('wav'):
            audio_splitter = SplitWavAudioMubin(directory.replace('raw', 'processed'), filename) # put into datasets/processed instead of datasets/raw
            audio_splitter.multiple_split(start_sec=start_sec, sec_per_split=audio_length, overlap=audio_overlap)  # ignore the first 1 second since it is blank


# Load and preprocess data
def load_fsdd_data(directory, max_length=50):
    data = defaultdict(list)
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith('wav'):
            team_name, speaker_id, clip_num = filename.split('_')
            filepath = os.path.join(directory, filename)
            _, sr = librosa.load(filepath, sr=None)  # just to get sr can change manually ig
            denoised_audio = denoise(filepath)
            # data augmentation n times
            for _ in range(10):
                roll_range = (0, 1000)
                rate_range = (0.8, 1.2)
                pitch_range = (-3, 3)
                augmented_audio = shift(denoised_audio, shift_range=roll_range)
                augmented_audio = stretch(augmented_audio, rate=rate_range)
                augmented_audio = pitch(augmented_audio, n_steps=pitch_range)

                mfccs = librosa.feature.mfcc(y=augmented_audio,sr=sr, n_mfcc=13)
                delta_mfccs = librosa.feature.delta(mfccs)
                delta2_mfccs = librosa.feature.delta(delta_mfccs)
                comprehensive_mfccs = np.concatenate((mfccs, delta_mfccs, delta2_mfccs))
                #Pad or truncate MFCCs to the fixed length
                if comprehensive_mfccs.shape[1] < max_length:
                    comprehensive_mfccs = np.pad(mfccs, ((0, 0), (0, max_length-comprehensive_mfccs.shape[1])), mode='constant')
                else:
                    comprehensive_mfccs = comprehensive_mfccs[:, :max_length] # og code was comprehensive_mfccs = comprehensive_mfccs[1, max_length] ??
                    
                
                data[team_name+speaker_id].append((comprehensive_mfccs, int(clip_num))) # eg. {'GPTEAM_member0001' : [(mfccs1, 1), (mfccs2, 2)]}
    
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


train_directory = "../datasets/processed/train"
val_directory = "../datasets/processed/val"
max_length = 150
train_data = load_fsdd_data(train_directory, max_length=max_length)
val_data = load_fsdd_data(val_directory, max_length=max_length)
        

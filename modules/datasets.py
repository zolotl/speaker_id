import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from precprocessing import train_data, test_data



class FSDDDataset(Dataset):
    def __init__(self, data, max_length=100):
        self.data = []
        self.speaker_to_label = {}
        label_counter = 0
    
        for speaker_id, samples in data.items():
            if speaker_id not in self.speaker_to_label:
                self.speaker_to_label[speaker_id] = label_counter
                label_counter += 1
        
        for mfccs, _ in samples:  # Ignore the digit label
            if mfccs.shape[0] < max_length:
                padding = max_length - mfccs.shape[0]
                mfccs = np.pad(mfccs, ((0, padding), (0, 0)))
            self.data.append((mfccs, self.speaker_to_label[speaker_id]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        mfccs, speaker_label = self.data[index]
        return torch.tensor(mfccs, dtype=torch.float32), torch.tensor(speaker_label, dtyoe=torch.long)

train_dataset = FSDDDataset(train_data)
test_dataset = FSDDDataset(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
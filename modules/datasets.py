import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from preprocessing import train_data, val_data


class FSDDDataset(Dataset):
    def __init__(self, data, speaker_to_label, max_length=50):
        self.data = []
        self.speaker_to_label = {}
        self.speaker_to_label = speaker_to_label
    
        for speaker_id, samples in data.items():
            for mfccs, _ in samples:  # Ignore the team_name label
                self.data.append((mfccs, self.speaker_to_label[speaker_id]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        mfccs, speaker_label = self.data[index]
        return torch.tensor(mfccs, dtype=torch.float32), torch.tensor(speaker_label, dtype=torch.long)


def create_speaker_label_map(data):
  label_counter = 0
  speaker_to_label = {}
  for speaker_id, _ in data.items():
      if speaker_id not in speaker_to_label:
          speaker_to_label[speaker_id] = label_counter
          label_counter += 1
  return speaker_to_label

speaker_label_map = create_speaker_label_map(train_data)

train_dataset = FSDDDataset(train_data, speaker_label_map)
val_dataset = FSDDDataset(val_data, speaker_label_map)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)


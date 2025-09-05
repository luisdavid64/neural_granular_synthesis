"""
    Dataloader for sonified data with tissue transitions
"""

import json
import torch
from torch.utils.data.dataset import Dataset
import os
import torchaudio

class BiosonixData(Dataset):
    def __init__(self, 
                 data_path,
                 hop_size=512, 
                 transform=None):
        """
        Args:
            data (list of np.array): List of audio samples.
            labels (list of int): List of corresponding labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = data_path['data']
        self.data = self.get_files(ext='wav')
        self.labels = self.get_files(ext='json', contains='_enhanced')
        self.hop_size = hop_size
        self.transform = transform

    def get_files(self, ext='mp4', contains=None):
        files = []
        for root, _, filenames in os.walk(self.data_path):
            for filename in filenames:
                if filename.endswith('.' + ext):
                    if contains is None or contains in filename:
                        files.append(os.path.join(root, filename))
        return files
    
    def add_start_end_sample_to_label(self, label, sr):
        for event in label:
            event['start_sample'] = int(event['start_s'] * sr)
            event['end_sample'] = int(event['end_s'] * sr)
        return label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample, sr = torchaudio.load(sample)
        with open(label, 'r') as label_file:
            label = json.load(label_file)
            label = self.add_start_end_sample_to_label(label, sr)

        if self.transform:
            sample = self.transform(sample)

        return sample, sr, label


# Test
if __name__ == "__main__":
    data_path = {'data': 'stopping_enhanced'}
    dataset = BiosonixData(data_path=data_path)
    print(f"Dataset size: {len(dataset)}")
    sample, sr, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Sample rate: {sr}, Label: {label}")
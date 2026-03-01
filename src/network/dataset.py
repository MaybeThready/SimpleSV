import librosa
import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from .config import DatasetConfig


class CNCelebLearnDataset(Dataset):
    """
    CNCeleb学习数据集
    """
    def __init__(self, data_dir: str, n_mels: int, sample_rate: int, window_length: float, hop_length: float, mfcc_select_length: float):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.mfcc_select_length = mfcc_select_length

        id_list = os.listdir(self.data_dir)
        records = []
        for idx, speaker_id in enumerate(id_list):
            speaker_dir = os.path.join(self.data_dir, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue
            file_list = os.listdir(speaker_dir)
            for file_name in file_list:
                file_path = os.path.join(speaker_dir, file_name)
                records.append(
                    {
                        "speaker_id": speaker_id,
                        "train_label": idx,
                        "file_path": file_path
                    }
                )

        self.data = pd.DataFrame(records)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        file_path = record["file_path"]
        train_label = record["train_label"]

        flac, _ = librosa.load(file_path, sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(y=flac, sr=self.sample_rate, n_mfcc=self.n_mels, n_fft=self.window_length, hop_length=self.hop_length)

        # 对MFCC进行随机裁剪或填充，使其长度固定
        if mfcc.shape[1] < self.mfcc_select_length:
            padding = np.zeros((mfcc.shape[0], self.mfcc_select_length - mfcc.shape[1]))
            mfcc = np.concatenate((mfcc, padding), axis=1)
        elif mfcc.shape[1] > self.mfcc_select_length:
            start_idx = np.random.randint(0, mfcc.shape[1] - self.mfcc_select_length)
            mfcc = mfcc[:, start_idx:start_idx + self.mfcc_select_length]

        # Cepstral Mean Normalisation
        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

        return torch.from_numpy(mfcc).float(), train_label


class CNCelebTestDataset(Dataset):
    """
    CNCeleb测试数据集
    """
    def __init__(self, data_dir: str, n_mels: int, sample_rate: int, window_length: int, hop_length: int):
        self.data_dir = data_dir
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.hop_length = hop_length

        file_names = os.listdir(self.data_dir)
        records = []
        for file_name in file_names:
            speaker_id = file_name.split("-")[0]
            file_path = os.path.join(self.data_dir, file_name)
            records.append(
                {
                    "speaker_id": speaker_id,
                    "file_path": file_path
                }
            )

        self.data = pd.DataFrame(records)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        record = self.data.iloc[idx]
        file_path = record["file_path"]
        speaker_id = record["speaker_id"]

        flac, _ = librosa.load(file_path, sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(y=flac, sr=self.sample_rate, n_mfcc=self.n_mels, n_fft=self.window_length, hop_length=self.hop_length)

        mfcc = mfcc - np.mean(mfcc, axis=1, keepdims=True)

        return torch.from_numpy(mfcc).float(), speaker_id


def build_datasets(config: DatasetConfig, split=True) -> tuple:
    learn_dir = os.path.join(config.root_dir, "data")
    enroll_dir = os.path.join(config.root_dir, r"eval\enroll")
    test_dir = os.path.join(config.root_dir, r"eval\test")

    learn_dataset = CNCelebLearnDataset(learn_dir, config.n_mels, config.sample_rate, config.window_length, config.hop_length, config.mfcc_select_length)
    enroll_dataset = CNCelebTestDataset(enroll_dir, config.n_mels, config.sample_rate, config.window_length, config.hop_length)
    test_dataset = CNCelebTestDataset(test_dir, config.n_mels, config.sample_rate, config.window_length, config.hop_length)

    if not split:
        return learn_dataset, enroll_dataset, test_dataset
    
    train_dataset_size = int(len(learn_dataset) * config.train_split_ratio)
    val_dataset_size = len(learn_dataset) - train_dataset_size
    train_dataset, val_dataset = torch.utils.data.random_split(learn_dataset, [train_dataset_size, val_dataset_size])

    return train_dataset, val_dataset, enroll_dataset, test_dataset

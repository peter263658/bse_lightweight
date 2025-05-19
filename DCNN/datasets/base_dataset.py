import torch
import torchaudio
import os

from pathlib import Path


SR = 16000
# N_MICROPHONE_SECONDS = 1
# N_MICS = 4


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 noisy_dataset_dir,
                 target_dataset_dir,
                 sr=SR,
                 mono=False):

        self.sr = sr
        self.target_dataset_dir = target_dataset_dir
        self.noisy_dataset_dir = noisy_dataset_dir

        self.mono = mono

        self.noisy_file_paths = self._get_file_paths(noisy_dataset_dir)
        self.target_file_paths = self._get_file_paths(target_dataset_dir)
      
    def __len__(self):
        return len(self.noisy_file_paths)

    # ── DCNN/datasets/base_dataset.py ─────────────────────────────
    def __getitem__(self, index):
        clean_path = self.target_file_paths[index]   # x
        noisy_path = self.noisy_file_paths[index]    # x + n

        clean, _ = torchaudio.load(clean_path)
        noisy, _ = torchaudio.load(noisy_path)

        # 依 LBCCN 資料生成規則，mixture = clean + noise
        noise = noisy - clean                       # n

        if self.mono:
            return noisy[0], clean[0], noise[0]     # (mix, voice_t, noise_t)
        else:
            return noisy, clean, noise


    def _get_file_paths(self, dataset_dir):
        # Convert to Path object if it's a string
        dataset_dir = Path(dataset_dir) if isinstance(dataset_dir, str) else dataset_dir
        # Just return the full paths directly
        file_paths = sorted(dataset_dir.rglob('*.wav'))
        return [str(fp) for fp in file_paths]  # Convert Path objects to strings

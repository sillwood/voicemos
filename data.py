import os

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

from tabula import DataFeature

INFLUENCE_CORRECTION = [
    'sys2dea8-uttaca68d6.wav',
    'sys86a3b-utt81bb889.wav',
]


def process_fn(list_path):
    if "TRAIN" in list_path:
        mean_csv_path = list_path.replace("TRAINSET", "train_mos_list.txt")
    else:
        mean_csv_path = list_path.replace("DEVSET", "val_mos_list.txt")
    with open(mean_csv_path, "r") as f:
        data = f.readlines()
    data = [i.rstrip().split(",") for i in data]

    mean_scores = {i: float(j) for i, j in data}

    with open(list_path, "r") as f:
        data = f.readlines()
    data = [i.rstrip().split(",") for i in data]

    base_dir = os.path.dirname(os.path.dirname(list_path))  # Get parent folder
    proc_data = []
    for row_data in data:
        listener_info = row_data[-1].split("_")
        wav_file = row_data[1]
        data_dict = {
            "system": row_data[0],
            "audio_path": os.path.join(base_dir, "wav", wav_file),
            "score": float(row_data[2]),
            "mean_score": mean_scores[wav_file],
            "listener": {
                "age_range": listener_info[1],
                "id": listener_info[2],
                "gender": listener_info[3],
                "impairment": listener_info[-1],
            },
        }
        proc_data.append(data_dict)

    return proc_data


def process_fn_mean(list_path, ood_path=None, inf_filter=False):
    mean_csv_path = list_path
    with open(mean_csv_path, "r") as f:
        data = f.readlines()
    data = [i.rstrip().split(",") for i in data]

    base_dir = os.path.dirname(os.path.dirname(mean_csv_path))  # Get parent folder
    proc_data = []
    for row_data in data:
        wav_file = row_data[0]
        if 'unlabeled_mos_list.txt' in mean_csv_path or "test.scp" in mean_csv_path:
            mean_score = 0.0
        else:
            mean_score = float(row_data[1])
        if inf_filter and row_data[0] in INFLUENCE_CORRECTION:
            continue
        data_dict = {
            "audio_path": os.path.join(base_dir, "wav", wav_file),
            "mean_score": mean_score,
        }
        proc_data.append(data_dict)

    return proc_data


class FilenameFeature(DataFeature):
    def feat(self, data):

        fpath = data["audio_path"]
        fname = os.path.basename(fpath)

        return fname

    def collate(self, batch_data):
        return batch_data


class ListenerIDFeature(DataFeature):
    def __init__(self):

        with open(
            "/home/jiameng/data_voicemos/phase1-main/DATA/listeners.yaml", "r"
        ) as f:
            listeners = yaml.load(f, Loader=yaml.FullLoader)
        self.listener_table = {
            listener_id: i for i, listener_id in enumerate(listeners["listeners"])
        }

    def feat(self, data):
        listener_id = self.listener_table[data["listener"]["id"]]
        return listener_id


class MeanScoreFeature(DataFeature):
    def feat(self, data):
        return data["mean_score"]


class Wav2vecFeature(DataFeature):
    def feat(self, data):
        fname = os.path.basename(data["audio_path"]).replace(".wav", ".npy")

        feat_path = os.path.join("wav2vec_feats", fname)

        feat = np.load(feat_path)
        return feat

    def collate(self, batch_data):
        new_batch = []

        max_length = max([i.shape[0] for i in batch_data])
        lengths = []
        for data in batch_data:
            length = data.shape[0]
            pad_length = max_length - length
            data = np.pad(data, ((0, pad_length), (0, 0)))
            new_batch.append(torch.Tensor(data))
            lengths.append(length)
        batch_data = torch.stack(new_batch)

        return {
            "data": batch_data,
            "lengths": lengths,
        }


class STFTFeature(DataFeature):
    def __init__(self, length_modulo=None):
        self.length_modulo = length_modulo

    def feat(self, data):

        fft_path = data["audio_path"].replace("/wav/", "/stft/").replace(".wav", ".npy")

        if os.path.exists(fft_path):
            spec = np.load(fft_path)
        else:
            wav, _ = librosa.load(data["audio_path"])
            spec = np.abs(librosa.stft(wav, n_fft=512)).T
            np.save(fft_path, spec)

        return spec

    def collate(self, batch_data):
        new_batch = []

        max_length = max([i.shape[0] for i in batch_data])
        if self.length_modulo is not None:
            max_length = max_length + (
                self.length_modulo - max_length % self.length_modulo
            )
        lengths = []
        for data in batch_data:
            length = data.shape[0]
            pad_length = max_length - length
            data = np.pad(data, ((0, pad_length), (0, 0)))
            new_batch.append(torch.Tensor(data))
            lengths.append(length)
        batch_data = torch.stack(new_batch)

        return {
            "data": batch_data,
            "lengths": lengths,
        }


class WavFeature(DataFeature):
    def __init__(self, length_modulo=None):

        self.sr = 16000
        self.length_modulo = length_modulo

    def feat(self, data):

        wav, _ = sf.read(data["audio_path"])

        return wav

    def collate(self, batch_data):
        new_batch = []

        max_length = max([len(i) for i in batch_data])
        if self.length_modulo is not None:
            max_length = max_length + (
                self.length_modulo - max_length % self.length_modulo
            )
        lengths = []
        for data in batch_data:
            length = len(data)
            pad_length = max_length - length
            data = np.pad(data, (0, pad_length), mode="edge")
            new_batch.append(data)
            lengths.append(length)
        batch_data = torch.tensor(np.array(new_batch)).float()
        lengths = torch.tensor(np.array(lengths)).float()

        return {
            "data": batch_data,
            "lengths": lengths,
        }


class ScoreFeature(DataFeature):
    def feat(self, data):
        score = data["score"]

        return score

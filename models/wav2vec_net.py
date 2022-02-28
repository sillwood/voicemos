import math

import fairseq
import torch
import torch.nn as nn

from utils import make_non_pad_mask, make_pad_mask


class MosPredictor(nn.Module):
    def __init__(self, cp_path):
        super().__init__()

        model, cfg = fairseq.checkpoint_utils.load_model_ensemble([cp_path])
        ssl_model = model[0]
        ssl_model.remove_pretraining_modules()

        ssl_model_type = cp_path.split("/")[-1]
        if ssl_model_type == "wav2vec_small.pt":
            ssl_out_dim = 768
        elif ssl_model_type in ["w2v_large_lv_fsh_swbd_cv.pt", "xlsr_53_56k.pt"]:
            ssl_out_dim = 1024
        else:
            raise ("*** ERROR *** SSL model type " + ssl_model_type + " not supported.")

        print("Loading checkpoint")
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.output_layer = nn.Linear(self.ssl_features, 1)

    def forward(self, wav, lengths=None):
        wav = wav.squeeze(1)  ## [batches, audio_len]
        if lengths is not None:
            # Not used in final model and only works after bug fixing Fairseq anyway
            wav_mask = (
                make_pad_mask(
                    lengths,
                    xs=wav,
                )
                .to(wav.device)
                .bool()
            )
            res = self.ssl_model(
                wav, padding_mask=wav_mask, mask=True, features_only=True
            )
        else:
            res = self.ssl_model(wav, mask=False, features_only=True)
        x = res["x"]
        x = torch.mean(x, 1)
        x = torch.sigmoid(self.output_layer(x)) * 4 + 1
        return x.squeeze(1)

# Overview

Entry to the 1st VoiceMOS challenge. This model is built on the SSL-MOS model introduced in [Generalization Ability of MOS Prediction Networks](https://github.com/nii-yamagishilab/mos-finetune-ssl) and introduces [SWA-Gaussian](https://github.com/wjmaddox/swa_gaussian) to the mix. There is also a notebook which allows for the calculation of [influence functions](https://arxiv.org/abs/1703.04730) of data points with respect to various test points.

This repo uses Tabula, a small work-in-progress boilerplate "package" to help simplify and speed up model development and research. The main features are Slates (which defines an epoch run), DataFeatures (a feature to use as input or ground truth) and Helpers (all the metric keeping and other utilities), all of which are designed to be plug-and-play. It's basically similar to PyTorch-Lightning but with a lot more flexibility in day-to-day use.

Apologies in advance for the immaturity of the codebase.

# Training

## Finetuning on wav2vec2.0

Download the `wav2vec_small.pt` checkpoint from the [Fairseq repo](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec) and put it at `checkpoints/fairseq/wav2vec_small.pt` (or set this in the yaml config files at model.cp_path)

Use the config files to also set the paths for where the data is.

Note that the `ssl_config.yaml` and `ood_config.yaml` files correspond to the main track and OOD track.

```
python train.py config=ssl_config.yaml exp_name=base-model
```

## Training with SWAG

The finetuning command with SWAG is as follows

```
python swag_train.py config=ssl_config.yaml checkpoint.path=checkpoints/base-model/iter_30000.pt
```

Or to use the OOD track configs

```
python swag_train.py config=ood_config.yaml checkpoint.path=checkpoints/base-model/iter_30000.pt
```

# Evaluation

Evaluation on the test set can be run as follows, where the config files can either be the ood or the main track configs

```
python eval.py config=YOUR_CONFIG.yaml checkpoint.path=checkpoints/YOUR_CHECKPOINT
```

Or for the dev set:

```
python eval.py config=YOUR_CONFIG.yaml checkpoint.path=checkpoints/YOUR_CHECKPOINT eval.set=dev
```

# Influence Functions notebook

This can be found at `notebooks/influence_functions.ipynb`

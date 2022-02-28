import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from data import FilenameFeature, MeanScoreFeature, WavFeature, process_fn_mean
from helpers import LRSchedulerHelper, MSEHelper, WandbHelper
from models.wav2vec_net import MosPredictor
from slates import EvalSlate, TrainSlate
from tabula import CheckpointHelper, DataLoader, Dataset, SubslateHelper


def main():
    cli_conf = OmegaConf.from_cli()
    with open(cli_conf.config, "r") as f:
        conf = OmegaConf.load(f)
    conf = OmegaConf.merge(conf, cli_conf)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)

    model = MosPredictor(**conf.model)
    model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f" Total number of parameters: {total_params}")

    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), **conf.optimizer, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    checkpoint_helper = CheckpointHelper(
        conf.exp_name,
        {
            "model": model,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        },
        save_epoch=conf.checkpoint.epoch,
        save_iters=conf.checkpoint.iters,
        only_save_last=False,
    )

    data_features = {
        "fnames": FilenameFeature(),
        "mean_score": MeanScoreFeature(),
        "wav": WavFeature(length_modulo=320),
    }
    train_set = Dataset(
        conf.training.data_path,
        data_features,
        proc_fn=partial(process_fn_mean, inf_filter=True),
    )
    train_loader = DataLoader(
        train_set, num_workers=8, shuffle=True, batch_size=conf.training.batch_size,
    )
    valid_set = Dataset(conf.eval.dev.data_path, data_features, proc_fn=process_fn_mean)
    valid_loader = DataLoader(
        valid_set, num_workers=8, shuffle=True, batch_size=conf.eval.batch_size,
    )

    model.train()

    # Helpers

    eval_slate = EvalSlate(
        model,
        helpers=[
            MSEHelper(),
        ],
    )
    helpers = [
        LRSchedulerHelper(lr_scheduler),
        SubslateHelper(
            eval_slate,
            valid_loader,
            run_epoch=conf.eval.run_epoch,
            run_iters=conf.eval.run_iters,
        ),
        checkpoint_helper,
        WandbHelper(OmegaConf.to_container(conf)),
    ]

    train_slate = TrainSlate(criterion, model, optimizer, helpers)

    if conf.checkpoint.path:
        metadata = checkpoint_helper.load(conf.checkpoint.path)
        train_slate.epoch = metadata["epoch"] + 1
        train_slate.iters = metadata["iters"] + 1

    train_slate.run(train_loader, max_epochs=conf.training.epochs)


if __name__ == "__main__":
    main()

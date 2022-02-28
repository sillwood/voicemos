import random
from functools import partial

import numpy as np
import torch
from omegaconf import OmegaConf

from data import FilenameFeature, MeanScoreFeature, WavFeature, process_fn_mean
from helpers import EvalSave, MSEHelper, SwagEvalHelper
from models.wav2vec_net import MosPredictor
from slates import EvalSlate
from swag.posteriors import SWAG
from tabula import CheckpointHelper, DataLoader, Dataset


def main():
    cli_conf = OmegaConf.from_cli()
    conf_file = cli_conf.config
    with open(conf_file, "r") as f:
        conf = OmegaConf.load(f)
    conf = OmegaConf.merge(conf, cli_conf)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    torch.cuda.manual_seed(conf.seed)

    model = MosPredictor(**conf.model)
    model.cuda()
    model.eval()
    swag_model = SWAG(
        MosPredictor,
        no_cov_mat=False,
        max_num_models=10,
        **conf.model,
    )
    swag_model.cuda()
    swag_model.eval()

    checkpoint_helper = CheckpointHelper(
        conf.exp_name,
        {
            "model": model,
            "swag_model": swag_model,
        },
        save_epoch=conf.checkpoint.epoch,
        save_iters=conf.checkpoint.iters,
    )

    data_features = {
        "fnames": FilenameFeature(),
        "wav": WavFeature(),
        "mean_score": MeanScoreFeature(),
    }

    eval_set = conf.eval[conf.eval.set]

    valid_set = Dataset(
        eval_set.data_path,
        data_features,
        proc_fn=partial(process_fn_mean, inf_filter=False),
    )
    valid_loader = DataLoader(
        valid_set, num_workers=8, shuffle=False, batch_size=conf.eval.batch_size
    )

    swag_helper = SwagEvalHelper(eval_set.out_file)
    eval_helpers = [
        MSEHelper(),
        EvalSave(eval_set.out_file),
    ]

    _ = checkpoint_helper.load(conf.checkpoint.path)

    if conf.eval.run_type == "swag":
        '''
        eval_slate = EvalSlate(model, eval_helpers)
        eval_slate.run(valid_loader, max_epochs=1)
        swag_mean_slate = EvalSlate(swag_model, eval_helpers)
        swag_model.sample(0.0)
        swag_mean_slate.run(valid_loader, max_epochs=1)
        '''
        swag_slate = EvalSlate(swag_model, [swag_helper])
        for _ in range(conf.eval.swag_samples):
            swag_model.sample(conf.eval.swag_scale, cov=True)
            swag_slate.run(valid_loader, max_epochs=1)

        swag_helper.compile_scores()
    elif conf.eval.run_type == "swag_mean":
        swag_mean_slate = EvalSlate(swag_model, eval_helpers)
        swag_model.sample(0.0)
        swag_mean_slate.run(valid_loader, max_epochs=1)
    else:
        eval_slate = EvalSlate(model, eval_helpers)
        eval_slate.run(valid_loader, max_epochs=1)


if __name__ == "__main__":
    main()

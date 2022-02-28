import os

import numpy as np
import scipy
import wandb

from tabula import Helper


class WandbHelper(Helper):
    def __init__(self, conf):

        wandb.init(project="voicemos", entity="jiamenggao", config=conf)
        wandb.run.name = conf["exp_name"]

    def iter_end(self, data, metadata):
        wandb.log({"loss": data["loss"]})


class LRSchedulerHelper(Helper):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def epoch_end(self, data, metadata):
        self.scheduler.step()


class SWAHelper(Helper):
    def __init__(self, swa_model, model, scheduler, swa_scheduler, swa_start):
        self.scheduler = scheduler
        self.model = model
        self.swa_scheduler = scheduler
        self.swa_model = swa_model
        self.swa_start = swa_start

    def epoch_end(self, data, metadata):
        if metadata["epoch"] > self.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        else:
            self.scheduler.step()


class SWAGHelper(Helper):
    def __init__(
        self, swa_model, model, scheduler, swa_start, swa_epoch=True, swa_iters=None
    ):
        self.scheduler = scheduler
        self.model = model
        self.swa_model = swa_model
        self.swa_start = swa_start
        self.swa_epoch = swa_epoch
        self.swa_iters = swa_iters

    def iter_end(self, data, metadata):
        if (
            self.swa_iters is not None
            and metadata["epoch"] > self.swa_start
            and metadata["iters"] % self.swa_iters == 0
        ):
            self.swa_model.collect_model(self.model)

    def epoch_end(self, data, metadata):
        if metadata["epoch"] > self.swa_start:
            if self.swa_epoch:
                self.swa_model.collect_model(self.model)
        else:
            self.scheduler.step()


class SwagSampleHelper(Helper):
    def __init__(self, swag_model):
        self.swag_model = swag_model

    def epoch_start(self, data, metadata):
        self.swag_model.sample(0.0)


class SwagEvalHelper(Helper):
    def __init__(self, out_fname):

        self.score_dict = {}
        self.out_fname = out_fname

    def epoch_start(self, data, metadata):
        self.current_scores = []

    def iter_end(self, data, metadata):
        gt_scores = data["inputs"]["mean_score"]
        pred_scores = data["outputs"]
        fnames = data["inputs"]["fnames"]

        for fname, pred_score, gt_score in zip(fnames, pred_scores, gt_scores):
            system = fname.split("-")[0]
            if fname not in self.score_dict:
                self.score_dict[fname] = [
                    {
                        "pred_score": pred_score.cpu().numpy(),
                        "gt_score": gt_score.cpu().numpy(),
                        "system": system,
                    }
                ]
            else:
                self.score_dict[fname].append(
                    {
                        "pred_score": pred_score.cpu().numpy(),
                        "gt_score": gt_score.cpu().numpy(),
                        "system": system,
                    }
                )

    def compile_scores(self):
        score_dict = {}
        for key, v in self.score_dict.items():
            pred_score = sum([i["pred_score"] for i in v]) / len(v)
            gt_score = sum([i["gt_score"] for i in v]) / len(v)
            system = v[0]["system"]
            score_dict[key] = {
                "pred_score": pred_score,
                "gt_score": gt_score,
                "system": system,
            }
        scores = [(v["pred_score"], v["gt_score"]) for k, v in score_dict.items()]
        scores = np.array(scores)
        pred_scores = scores[:, 0]
        gt_scores = scores[:, 1]

        sys_dict = {}
        systems = list(set([v["system"] for v in score_dict.values()]))
        for system in systems:
            scores = [
                (v["pred_score"], v["gt_score"])
                for k, v in score_dict.items()
                if v["system"] == system
            ]
            scores = np.array(scores)
            pred_score = np.mean(scores[:, 0])
            gt_score = np.mean(scores[:, 1])

            sys_dict[system] = {
                "pred_score": pred_score,
                "gt_score": gt_score,
            }

        scores = [(v["pred_score"], v["gt_score"]) for k, v in sys_dict.items()]
        scores = np.array(scores)
        sys_pred_scores = scores[:, 0]
        sys_gt_scores = scores[:, 1]

        utt_scores = [
            np.mean((gt_scores - pred_scores) ** 2),
            np.corrcoef(gt_scores, pred_scores)[0][1],
            scipy.stats.kendalltau(gt_scores, pred_scores)[0],
            scipy.stats.spearmanr(gt_scores, pred_scores)[0],
        ]
        sys_scores = [
            np.mean((sys_gt_scores - sys_pred_scores) ** 2),
            np.corrcoef(sys_gt_scores, sys_pred_scores)[0][1],
            scipy.stats.kendalltau(sys_gt_scores, sys_pred_scores)[0],
            scipy.stats.spearmanr(sys_gt_scores, sys_pred_scores)[0],
        ]
        row = "{:>12} {:>10} {:>10} {:>10} {:>10}"

        utt_scores = ["{:.4f}".format(i) for i in utt_scores]
        sys_scores = ["{:.4f}".format(i) for i in sys_scores]
        print(row.format("", "MSE", "LCC", "KTAU", "SRCC"))
        print(row.format("Utterance", *utt_scores))
        print(row.format("System", *sys_scores))

        with open(self.out_fname, "w") as f:
            for fname, output in score_dict.items():
                score = output["pred_score"]
                f.write(f"{fname},{score}\n")


class EvalSave(Helper):
    def __init__(self, out_fname):
        self.out_fname = out_fname
        with open(self.out_fname, "w") as _:
            pass

    def iter_end(self, data, metadata):
        # Need a better writer
        with open(self.out_fname, "a") as f:
            for fname, output in zip(data["inputs"]["fnames"], data["outputs"]):
                score = output.item()
                f.write(f"{fname},{score}\n")


class FeatSave(Helper):
    def iter_end(self, data, metadata):
        # Need a better writer
        for fname, feat in zip(data["inputs"]["fnames"], data["feats"]):
            fname = fname.replace(".wav", ".npy")

            feat_path = os.path.join("wav2vec_feats", fname)

            np.save(feat_path, feat.cpu().numpy())


class MSEHelper(Helper):
    def __init__(self):
        self.score_dict = {}

    def epoch_start(self, data, metadata):
        pass

    def iter_end(self, data, metadata):
        gt_scores = data["inputs"]["mean_score"]
        pred_scores = data["outputs"]
        fnames = data["inputs"]["fnames"]

        for fname, pred_score, gt_score in zip(fnames, pred_scores, gt_scores):
            system = fname.split("-")[0]
            self.score_dict[fname] = {
                "pred_score": pred_score.cpu().numpy(),
                "gt_score": gt_score.cpu().numpy(),
                "system": system,
            }

    def epoch_end(self, data, metadata):
        scores = [(v["pred_score"], v["gt_score"]) for k, v in self.score_dict.items()]
        scores = np.array(scores)
        pred_scores = scores[:, 0]
        gt_scores = scores[:, 1]

        sys_dict = {}
        for system in self._systems:
            scores = [
                (v["pred_score"], v["gt_score"])
                for k, v in self.score_dict.items()
                if v["system"] == system
            ]
            scores = np.array(scores)
            pred_score = np.mean(scores[:, 0])
            gt_score = np.mean(scores[:, 1])

            sys_dict[system] = {
                "pred_score": pred_score,
                "gt_score": gt_score,
            }

        scores = [(v["pred_score"], v["gt_score"]) for k, v in sys_dict.items()]
        scores = np.array(scores)
        sys_pred_scores = scores[:, 0]
        sys_gt_scores = scores[:, 1]

        utt_scores = [
            np.mean((gt_scores - pred_scores) ** 2),
            np.corrcoef(gt_scores, pred_scores)[0][1],
            scipy.stats.kendalltau(gt_scores, pred_scores)[0],
            scipy.stats.spearmanr(gt_scores, pred_scores)[0],
        ]
        sys_scores = [
            np.mean((sys_gt_scores - sys_pred_scores) ** 2),
            np.corrcoef(sys_gt_scores, sys_pred_scores)[0][1],
            scipy.stats.kendalltau(sys_gt_scores, sys_pred_scores)[0],
            scipy.stats.spearmanr(sys_gt_scores, sys_pred_scores)[0],
        ]
        row = "{:>12} {:>10} {:>10} {:>10} {:>10}"

        utt_scores = ["{:.4f}".format(i) for i in utt_scores]
        sys_scores = ["{:.4f}".format(i) for i in sys_scores]
        print(row.format("", "MSE", "LCC", "KTAU", "SRCC"))
        print(row.format("Utterance", *utt_scores))
        print(row.format("System", *sys_scores))
        if wandb.run is not None:
            wandb.log(
                {
                    "Sys MSE": float(sys_scores[0]),
                    "Sys SRCC": float(sys_scores[-1]),
                    "Utt MSE": float(utt_scores[0]),
                    "Utt SRCC": float(utt_scores[-1]),
                }
            )

    @property
    def _systems(self):
        systems = list(set([v["system"] for v in self.score_dict.values()]))

        return systems

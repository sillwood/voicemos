import torch
from torch.cuda.amp import GradScaler, autocast

from tabula import Slate


class TrainSlate(Slate):
    def __init__(
        self,
        loss_fn,
        model,
        optimizer,
        helpers,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.helpers = helpers

        self.scaler = GradScaler()

    def step(self, batch_data):

        self.model.zero_grad()

        with autocast(dtype=torch.bfloat16):
            pred_mu = self.model(
                batch_data["wav"]["data"],
                # batch_data['wav']['lengths'],
                # feats=batch_data['feats']['data'],
            )

            loss = self.loss_fn(
                pred_mu,
                batch_data["mean_score"],
                # batch_data['feats']['lengths'],
            )

        # loss.backward()
        # self.optimizer.step()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_dict = {
            "Loss": loss.item(),
        }
        data = {"loss": loss.item()}

        return loss_dict, data


class GradAccumTrainSlate(Slate):
    def __init__(
        self,
        loss_fn,
        model,
        optimizer,
        helpers,
    ):
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.helpers = helpers

        self.scaler = GradScaler()

    def step(self, batch_data):

        if self.iters % 2 == 0:
            self.model.zero_grad()

        with autocast(dtype=torch.bfloat16):
            pred_mu = self.model(
                batch_data["wav"]["data"],
                # batch_data['wav']['lengths'],
                # feats=batch_data['feats']['data'],
            )

            loss = self.loss_fn(
                pred_mu,
                batch_data["mean_score"],
                # batch_data['feats']['lengths'],
            )

            loss = loss / 2

        # loss.backward()
        # self.optimizer.step()
        self.scaler.scale(loss).backward()
        if self.iters % 2 == 1:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        loss_dict = {
            "Loss": loss.item(),
        }
        data = {"loss": loss.item()}

        return loss_dict, data


class EvalSlate(Slate):
    train = False

    def __init__(
        self,
        model,
        helpers=None,
    ):
        super().__init__()

        self.model = model
        self.helpers = helpers

    def step(self, batch_data):

        with torch.no_grad():
            score = self.model(
                batch_data["wav"]["data"],
            )

        loss_dict = {}
        data = {"inputs": batch_data, "outputs": score}

        return loss_dict, data


class GenFeatsSlate(Slate):
    train = False

    def __init__(
        self,
        model,
        helpers=None,
    ):
        super().__init__()

        self.model = model
        self.helpers = helpers

    def step(self, batch_data):

        with torch.no_grad():
            feats = self.model.ssl_model(batch_data["wav"]["data"], mask=False, features_only=True)['x']

        loss_dict = {}
        data = {
            "inputs": batch_data,
            "feats": feats,
        }

        return loss_dict, data

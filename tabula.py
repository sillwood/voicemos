import os
from functools import partial
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils import data
from tqdm import tqdm


def _to_gpu(data):
    if isinstance(data, list):
        return [_to_gpu(i) for i in data]
    elif isinstance(data, dict):
        return {k: _to_gpu(v) for k, v in data.items()}
    elif isinstance(data, tuple):
        return tuple([_to_gpu(i) for i in data])
    elif torch.is_tensor(data):
        return data.cuda()
    else:
        return data


class Slate():
    epoch = None
    iters = None
    helpers = []
    train = True

    def __init__(self):

        self.epoch = 1
        self.iters = 0
        self._stop = False
        self._data = None

    def step(self, data):

        raise NotImplementedError

    def run(self, dataloader, max_epochs: Optional[int] = None,
            max_iters: Optional[int] = None):
        self._run(dataloader, max_epochs, max_iters)

    def _run(self, dataloader, max_epochs, max_iters):
        self._stop = False

        while not self._stop:
            if self.helpers is not None:
                for helper in self.helpers:
                    helper.epoch_start(data=self.data, metadata=self.metadata)

            data_enum = tqdm(dataloader)
            for batch_data in data_enum:
                if self.helpers is not None:
                    for helper in self.helpers:
                        helper.iter_start(data=self.data, metadata=self.metadata)

                batch_data = _to_gpu(batch_data)

                loss_dict, self._data = self.step(batch_data)

                message = [f"{k}: {v:.6f}" for k, v in loss_dict.items()]
                message = ", ".join(message)
                message = f"Epoch {self.epoch} Iter {self.iters} " + message
                data_enum.set_description(message)

                if self.helpers is not None:
                    for helper in self.helpers:
                        helper.iter_end(data=self.data, metadata=self.metadata)

                self.iters += 1

                if max_iters is not None and self.iters > max_iters:
                    self._stop = True
                    break

            if self.helpers is not None:
                for helper in self.helpers:
                    helper.epoch_end(data=self.data, metadata=self.metadata)

            self.epoch += 1
            if max_epochs is not None and self.epoch > max_epochs:
                self._stop = True

        self.shutdown()

    def shutdown(self):
        pass

    @property
    def data(self):
        return self._data

    @property
    def metadata(self):
        return {
            'epoch': self.epoch,
            'iters': self.iters,
        }


class DataFeature():

    def __init__(self):
        pass

    def feat(self, data):

        return data

    def collate(self, data):
        data = torch.tensor(data)
        return data


class Dataset(data.Dataset):

    def __init__(self, dataset, features, proc_fn=None):

        if proc_fn is not None:
            dataset = proc_fn(dataset)
        self.dataset = dataset
        self.features = features


    def __getitem__(self, idx):
        data = {key: feature.feat(self.dataset[idx]) for key, feature in self.features.items()}

        return data

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch, features):
    data = {key: feature.collate([data[key] for data in batch])
            for key, feature in features.items()}

    return data


class DataLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):

        _collate_fn = partial(collate_fn, features=dataset.features)
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class Helper():

    def epoch_start(self, data, metadata):
        pass

    def epoch_end(self, data, metadata):
        pass

    def iter_start(self, data, metadata):
        pass

    def iter_end(self, data, metadata):
        pass


class CheckpointHelper(Helper):
    save_dir = Path("checkpoints")

    def __init__(self, exp_name, checkpoint_dict, save_epoch=False, save_iters=None, only_save_last=True):

        assert save_iters is None or isinstance(save_iters, int), "save_iters must be None or int"

        self.exp_name = exp_name
        self.checkpoint_dict = checkpoint_dict
        self.save_epoch = save_epoch
        self.save_iters = save_iters
        self.only_save_last = only_save_last

    def epoch_end(self, data, metadata):
        if self.save_epoch:
            if self.only_save_last:
                fname = "lastmodel.pt"
            else:
                fname = f"epoch_{metadata['epoch']}.pt"
            self._save(data, metadata, fname)

    def iter_end(self, data, metadata):
        iteration = metadata['iters']
        if self.save_iters is not None and iteration > 0 and iteration % self.save_iters == 0:
            fname = f'iter_{iteration}.pt'
            self._save(data, metadata, fname)

    def _save(self, data, metadata, fname):
        save_dict = {
            'metadata': metadata,
            'checkpoint': {k: v.state_dict() for k, v in self.checkpoint_dict.items()},
        }
        checkpoint_path = self.checkpoint_dir / fname
        print(f"Saving checkpoint at {checkpoint_path}")
        torch.save(save_dict, checkpoint_path)

    def load(self, checkpoint_path):

        assert os.path.isfile(checkpoint_path)

        loaded_dict = torch.load(checkpoint_path, map_location='cpu')

        for k, v in self.checkpoint_dict.items():
            try:
                v.load_state_dict(loaded_dict['checkpoint'][k])
            except KeyError as e:
                print(e)

        return loaded_dict['metadata']

    @property
    def checkpoint_dir(self):
        checkpoint_dir = self.save_dir / self.exp_name
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        return checkpoint_dir


class SubslateHelper(Helper):
    def __init__(self, slate, dataloader, run_epoch=False, run_iters=None):

        assert run_iters is None or isinstance(run_iters, int), "save_iters must be None or int"

        self.slate = slate
        self.dataloader = dataloader
        self.run_epoch = run_epoch
        self.run_iters = run_iters

    def _run(self):
        if not self.slate.train:
            self.slate.model.eval()
        self.slate.run(self.dataloader, max_epochs=1)
        if not self.slate.train:
            self.slate.model.train()

    def epoch_end(self, data, metadata):
        if self.run_epoch:
            self._run()

    def iter_end(self, data, metadata):
        iteration = metadata['iters']
        if self.run_iters is not None and iteration > 0 and iteration % self.run_iters == 0:
            self._run()

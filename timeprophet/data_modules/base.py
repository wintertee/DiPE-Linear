from typing import List, Type

import pandas as pd
import pytorch_lightning as L
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):

    def __init__(self, data_x: torch.Tensor, data_y: torch.Tensor,
                 input_len: int, output_len: int):

        assert len(data_x) == len(data_y)

        self.data_x = data_x
        self.data_y = data_y
        self.input_len = input_len
        self.output_len = output_len

    def __len__(self) -> int:
        return len(self.data_x) - self.input_len - self.output_len + 1

    def __getitem__(self, idx) -> tuple[torch.Tensor]:
        return (self.data_x[idx:idx + self.input_len],
                self.data_y[idx + self.input_len:idx + self.input_len +
                            self.output_len])


class TimeSeriesDataModule(L.LightningDataModule):
    """This is a Dataset base class for time series data.

    """

    _all_features_num = None

    def __init__(self,
                 dataset_path: str,
                 batch_size: int,
                 input_len: int,
                 output_len: int,
                 x_features: List[int] | None = None,
                 y_features: List[int] | None = None,
                 all_features_num: int | None = None,
                 preprocessor: Type[BaseEstimator] = StandardScaler,
                 pin_memory: bool = True,
                 num_workers: int = 0,
                 persistent_workers: bool = False,
                 gpu: bool = False,
                 train_proportion: float = 1.0,
                 down_sampling: int = 1):

        super().__init__()

        if x_features is None or y_features is None:
            assert all_features_num is not None

        self.dataset_path = dataset_path
        self.input_len = input_len
        self.output_len = output_len
        self.x_features = x_features
        self.y_features = y_features
        self.all_features_num = all_features_num
        self.preprocessor = preprocessor()
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.gpu = gpu
        self.train_proportion = train_proportion
        self.down_sampling = down_sampling

        if self.gpu:
            self.pin_memory = False
            self.num_workers = 0
            self.persistent_workers = False

        if self.x_features is not None:
            self.x_features_num = len(self.x_features)
        else:
            self.x_features_num = self.all_features_num

        if self.y_features is not None:
            self.y_features_num = len(self.y_features)
        else:
            self.y_features_num = self.all_features_num

        self.is_setup = False

    def __read_data__(self) -> pd.DataFrame:
        raise NotImplementedError

    def __split_data__(self, all_data: pd.DataFrame) -> tuple[pd.DataFrame]:
        raise NotImplementedError

    def prepare_data(self) -> None:

        # read data in DataFrame Format
        all_data = self.__read_data__()

        train_data, val_data, test_data = self.__split_data__(all_data)

        # downsampling
        train_data = train_data.iloc[::self.down_sampling]
        val_data = val_data.iloc[::self.down_sampling]
        test_data = test_data.iloc[::self.down_sampling]

        # covert data to numpy array by sklearn preprocessor
        train_data = self.preprocessor.fit_transform(train_data)
        val_data = self.preprocessor.transform(val_data)
        test_data = self.preprocessor.transform(test_data)

        # convert data to float32 tensor
        train_data = torch.from_numpy(train_data).float()
        val_data = torch.from_numpy(val_data).float()
        test_data = torch.from_numpy(test_data).float()

        train_len = train_data.shape[0]
        train_data = train_data[:int(self.train_proportion * train_len), :]

        if self.gpu:
            train_data = train_data.cuda()
            val_data = val_data.cuda()
            test_data = test_data.cuda()

        if self.x_features is None:
            self.train_x = train_data
            self.val_x = val_data
            self.test_x = test_data
        else:
            self.train_x = train_data[:, self.x_features]
            self.val_x = val_data[:, self.x_features]
            self.test_x = test_data[:, self.x_features]

        if self.y_features is None:
            self.train_y = train_data
            self.val_y = val_data
            self.test_y = test_data
        else:
            self.train_y = train_data[:, self.y_features]
            self.val_y = val_data[:, self.y_features]
            self.test_y = test_data[:, self.y_features]

    def setup(self, stage: str) -> None:

        if not self.is_setup:
            self.train_dataset = TimeSeriesDataset(
                self.train_x,
                self.train_y,
                self.input_len,
                self.output_len,
            )
            self.val_dataset = TimeSeriesDataset(
                self.val_x,
                self.val_y,
                self.input_len,
                self.output_len,
            )
            self.test_dataset = TimeSeriesDataset(
                self.test_x,
                self.test_y,
                self.input_len,
                self.output_len,
            )
            self.is_setup = True

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

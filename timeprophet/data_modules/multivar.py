import pandas as pd

from .base import TimeSeriesDataModule


class MultivarDataModule(TimeSeriesDataModule):
    """DataModule for Multivariate Time series Datasets.

    For further information, please refer to the following repository:
    https://github.com/laiguokun/multivariate-time-series-data
    """

    def __read_data__(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset_path).drop('date', axis=1)

    def __split_data__(self, all_data: pd.DataFrame) -> tuple[pd.DataFrame]:
        """split the data into training, validation  and test sets.

        we follow the method used in Informer:
        https://github.com/zhouhaoyi/Informer2020/blob/0ac81e04d4095ecb97a3a78c7b49c936d8aa9933/data/data_loader.py#L50
        https://github.com/zhouhaoyi/Informer2020/blob/0ac81e04d4095ecb97a3a78c7b49c936d8aa9933/data/data_loader.py#L136
        """

        train_len = int(len(all_data) * 0.7)
        test_len = int(len(all_data) * 0.2)
        val_len = len(all_data) - train_len - test_len

        train_data = all_data[:train_len]
        val_data = all_data[train_len - self.input_len:train_len + val_len]
        test_data = all_data[train_len + val_len - self.input_len:train_len +
                             val_len + test_len]

        return train_data, val_data, test_data

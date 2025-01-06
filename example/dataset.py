"""Modal-agnostic DataModule."""

import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import wget
from lightning import LightningDataModule
from numpy.random import MT19937, Generator
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, StackDataset
from tqdm import tqdm


class Transform:
    """Base class for tensor transform."""

    def __call__(self, tensor: Tensor) -> Tensor:
        """Apply transform."""
        raise NotImplementedError


def split_path_list(path_list: list[Path], train_ratio: float = 0.8) -> tuple[list[Path], list[Path]]:
    """
    Split the path list into train and test.

    Parameters
    ----------
    path_list : list[Path]
        List of file paths.
    train_ratio : float
        Ratio of train data.

    Returns
    -------
    tuple[list[Path], list[Path]]
        Train and test path list.
    """
    train_len = int(len(path_list) * train_ratio)
    return path_list[:train_len], path_list[train_len:]


def load_tensor(path: Path) -> Tensor:
    """
    Load tensor from file(.npy, .pt).

    Parameters
    ----------
    path : Path
        File path.

    Returns
    -------
    Tensor
        Loaded tensor.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    """
    if path.suffix == ".npy":
        return Tensor(np.load(path))
    if path.suffix == ".pt" and isinstance(tensor := torch.load(path, weights_only=False), Tensor):
        return tensor
    msg = f"Unknown file extension: {path.suffix}"
    raise ValueError(msg)


class EpisodeDataset(Dataset[Tensor]):
    """
    Dataset for single modality data.

    Parameters
    ----------
    path_list : list[Path]
        List of file paths.
    transform : Transform
        Transform function.
    """

    def __init__(self, path_list: list[Path], transform: Transform) -> None:
        super().__init__()
        self.path_list = path_list
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of data.

        Returns
        -------
        int
            Number of data(Len of path_list).
        """
        return len(self.path_list)

    def __getitem__(self, idx: int) -> Tensor:
        """
        Get the data at the index and apply the transform.

        Parameters
        ----------
        idx : int
            Index of the data.

        Returns
        -------
        Tensor
            Transformed data.
        """
        return self.transform(load_tensor(self.path_list[idx]))


@dataclass
class DataConfig:
    """Single modal(action, observation, ...) data configuration."""

    prefix: str
    preprocess: Transform
    train_transform: Transform
    val_transform: Transform
    test_transform: Transform


@dataclass
class DataModuleConfig:
    """Configuration for EpisodeDataModule."""

    data_name: str
    processed_data_name: str
    batch_size: int
    num_workers: int
    gdrive_id: str
    train_ratio: float
    data_defs: tuple[DataConfig, ...]

    @property
    def data_dir(self) -> Path:
        """Path to the data directory."""
        return Path("data") / self.data_name

    @property
    def processed_data_dir(self) -> Path:
        """Path to the processed data directory."""
        return Path("data") / self.processed_data_name

    def load_from_gdrive(self) -> None:
        """Download data from Google Drive."""
        url = f"https://drive.usercontent.google.com/download?export=download&confirm=t&id={self.gdrive_id}"
        filename = Path("tmp.tar.gz")
        wget.download(url, str(filename))
        with tarfile.open(filename, "r:gz") as f:
            f.extractall(path=Path("data"), filter="data")
        Path(filename).unlink(missing_ok=False)


class EpisodeDataModule(LightningDataModule):
    """
    Modal-Agnostic DataModule.

    Train/Val/Test dataloaders yields [modal1(input), modal1(target), modal2(input), modal2(target), ...].
    """

    def __init__(self, config: DataModuleConfig) -> None:
        super().__init__()
        self.config = config

    def prepare_data(self) -> None:
        """Save processed data to `{data_name}_processed_episode` directory."""
        if not self.config.data_dir.exists():
            self.config.load_from_gdrive()

        if self.config.processed_data_dir.exists():
            return

        self.config.processed_data_dir.mkdir(parents=True, exist_ok=True)

        for data_type in self.config.data_defs:
            for path in tqdm(sorted(self.config.data_dir.glob(data_type.prefix))):
                tensor = data_type.preprocess(load_tensor(path))
                new_path = self.config.processed_data_dir / f"{path.stem}.pt"
                torch.save(tensor.detach().clone(), new_path)

    def setup(self, stage: str = "fit") -> None:  # noqa: ARG002
        """Create datasets."""
        train_dataset_list, val_dataset_list, test_dataset_list = [], [], []
        for data_type in self.config.data_defs:
            path_list = sorted(self.config.processed_data_dir.glob(data_type.prefix))
            train_path_list, val_path_list = split_path_list(path_list, self.config.train_ratio)
            train_dataset_list.append(EpisodeDataset(train_path_list, data_type.train_transform))
            val_dataset_list.append(EpisodeDataset(val_path_list, data_type.val_transform))

            test_path_list = []
            for train_path, val_path in zip(train_path_list, val_path_list, strict=False):
                test_path_list.extend([train_path, val_path])

            test_dataset_list.append(EpisodeDataset(test_path_list, data_type.test_transform))

        self.train_dataset = StackDataset(*train_dataset_list)
        self.val_dataset = StackDataset(*val_dataset_list)
        self.test_dataset = StackDataset(*test_dataset_list)

    def train_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """
        TrainDataLoader.

        Returns
        -------
        DataLoader[tuple[Tensor, ...]]
            DataLoader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """
        ValidationDataLoader.

        Returns
        -------
        DataLoader[tuple[Tensor, ...]]
            DataLoader. Shuffle is False.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )

    def test_dataloader(self) -> DataLoader[tuple[Tensor, ...]]:
        """
        TestDataLoader.

        Returns
        -------
        DataLoader[tuple[Tensor, ...]]
            DataLoader. Shuffle is False.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            persistent_workers=True,
            prefetch_factor=1,
        )


class Compose(Transform):
    def __init__(self, transforms: list[Callable]) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, data: Tensor) -> Tensor:
        for transform in self.transforms:
            data = transform(data)
        return data


class ObservationEncoder(Transform):
    """Encode video tensor into tokens."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        local_dir = Path(".model_cache") / model_name
        encoder_path = local_dir / "encoder.jit"
        if not encoder_path.exists():
            local_dir.mkdir(exist_ok=True, parents=True)
            url = f"https://huggingface.co/nvidia/{model_name}/resolve/main/encoder.jit"
            wget.download(url, str(encoder_path))
        self.encoder = torch.jit.load(encoder_path).eval()  # type: ignore[no-untyped-call]

    def __call__(self, video: Tensor) -> Tensor:
        """
        Tokenize video.

        Parameters
        ----------
        video : Tensor
            Video data. Shape: [L C H W].

        Returns
        -------
        Tensor
            Codes. Shape: [L C H' W'].
        """
        self.encoder.to("cuda")
        video = video.to("cuda")
        features, _ = self.encoder(video)
        return features.to("cpu")  # type: ignore[no-any-return]


class NormalizeAction(Transform):
    """
    Normalize 3D+ tensor with given max and min values.

    Parameters
    ----------
    max_array : list[float]
        Max values of the tensor along the last dim.

    min_array : list[float]
        Min values of the tensor along the last dim.
    """

    def __init__(self, max_array: list[float], min_array: list[float]) -> None:
        super().__init__()
        self.max_array = Tensor(max_array)
        self.min_array = Tensor(min_array)

    def __call__(self, data: Tensor) -> Tensor:
        """
        Apply normalization.

        Parameters
        ----------
        data : Tensor
            Data to be normalized. Shape: [batch*, dim].

        Returns
        -------
        Tensor
            Normalized data.
        """
        copy_data = data.detach().clone()
        copy_data -= self.min_array
        copy_data *= 1.0 / (self.max_array - self.min_array)
        copy_data *= 2.0
        copy_data += -1.0
        return copy_data


class RandomWindow(Transform):
    """Randomly slice sequence data."""

    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.randgen = Generator(MT19937(42))

    def __call__(self, data: Tensor) -> Tensor:
        """
        Select start idx with `randgen2` and slice data.

        Parameters
        ----------
        data : Tensor
            Sequence data to be sliced. Shape: [seq_len, *].

        Returns
        -------
        Tensor
            Sliced data. Shape: [window_size, *].
        """
        seq_len = data.shape[0]
        start_idx = self.randgen.integers(0, seq_len - self.window_size)
        return data[start_idx : start_idx + self.window_size]

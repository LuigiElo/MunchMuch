from pathlib import Path
import os

import typer
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import requests
from tqdm import tqdm

import kagglehub
from kagglehub import KaggleDatasetAdapter


class MyDataset(Dataset):
    """Munch paintings dataset."""

    def __init__(self, data_path: Path, transform=None) -> None:
        self.data_path = Path(__file__).parent.parent.parent / data_path
        self.transform = transform
        self.csv_path = self.data_path / "edvard_munch.csv"
        print(f"Looking for dataset at {self.csv_path}...")

        if self.csv_path.exists():
            print(f"Loading dataset from {self.csv_path}...")
            self.df = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        img_path = self.data_path / "munch_paintings" / row["filename"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = row.get("period", "unknown")

        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess raw Kaggle dataset into clean format."""
        ### get current root directory
        root_dir = Path(__file__).parent.parent.parent 
        raw_data_path = root_dir / self.data_path
        

        print("Downloading dataset from KaggleHub...")
        dataset_path = kagglehub.dataset_download(
            "isaienkov/edvard-munch-paintings",
            output_dir=raw_data_path
        )

        print(f"Dataset downloaded to: {dataset_path}")
        return dataset_path


def preprocess(data_path : Path = "data/raw", output_folder : Path = "data/processed") -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)

    dataset.__getitem__(0)  # Test loading the first item

if __name__ == "__main__":
    typer.run(preprocess)
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
        self.data_path = data_path
        self.transform = transform


        self.data_path = Path(data_path)
        self.csv_path = self.data_path / "clean_dataset.csv"

        if self.csv_path.exists():
            self.df = pd.read_csv(self.csv_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        img_path = row["image_path"]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = row.get("period", "unknown")

        return image, label

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess raw Kaggle dataset into clean format."""
        data_path = Path(self.data_path)
        output_folder = Path(output_folder)


        output_folder.mkdir(parents=True, exist_ok=True)
        images_folder = output_folder / "images"
        images_folder.mkdir(exist_ok=True)

        print("Downloading dataset from KaggleHub...")
        dataset_path = kagglehub.dataset_download(
            "isaienkov/edvard-munch-paintings"
        )

        print(f"Dataset downloaded to: {dataset_path}")

        # Try to locate CSV automatically
        csv_path = None
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(".csv"):
                    csv_path = Path(root) / f
                    break

        if csv_path is None:
            raise FileNotFoundError("No CSV file found in dataset")

        print(f"Using CSV: {csv_path}")

        df = pd.read_csv(csv_path)

        # Try to detect image column
        possible_img_cols = ["image", "image_url", "url", "filename"]
        img_col = None
        for col in possible_img_cols:
            if col in df.columns:
                img_col = col
                break

        if img_col is None:
            raise ValueError("No image column found in dataset")

        print(f"Using image column: {img_col}")


        valid_rows = []

        print("Downloading and validating images...")

        for i, row in tqdm(df.iterrows(), total=len(df)):
            img_value = row[img_col]
            save_path = images_folder / f"{i}.jpg"

            try:
                # Case 1: URL
                if str(img_value).startswith("http"):
                    img_data = requests.get(img_value, timeout=10).content
                    with open(save_path, "wb") as f:
                        f.write(img_data)

                # Case 2: local file path inside dataset
                else:
                    possible_path = Path(dataset_path) / img_value
                    if possible_path.exists():
                        Image.open(possible_path).convert("RGB").save(save_path)
                    else:
                        continue

                # Validate image
                img = Image.open(save_path).convert("RGB")
                img.verify()

                valid_rows.append(i)

            except Exception:
                continue

        print(f"Valid images: {len(valid_rows)} / {len(df)}")

        df = df.iloc[valid_rows].reset_index(drop=True)

        # Add image paths
        df["image_path"] = df.index.map(
            lambda i: str(images_folder / f"{i}.jpg")
        )

        # Add period label (based on year if available)
        if "year" in df.columns:

            def year_to_period(year):
                try:
                    year = int(year)
                    if year < 1890:
                        return "early"
                    elif year < 1905:
                        return "middle"
                    else:
                        return "late"
                except Exception:
                    return "unknown"

            df["period"] = df["year"].apply(year_to_period)

        else:
            df["period"] = "unknown"

        # Save clean dataset
        clean_csv_path = output_folder / "clean_dataset.csv"
        df.to_csv(clean_csv_path, index=False)

        print(f"Saved cleaned dataset to: {clean_csv_path}")


def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess("data/raw","data/processed"))
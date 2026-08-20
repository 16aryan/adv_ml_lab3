from pathlib import Path

from loguru import logger
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import typer

from lab3.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def split_sets_random(X, y, test_ratio=0.2, val_ratio=0.2, random_state=42):
    """
    Split the dataset into training, validation, and testing sets as Numpy arrays.
    """
    # Convert to numpy arrays if pandas objects are passed
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    if hasattr(y, "to_numpy"):
        y = y.to_numpy()
        
    # Split into train_val and test first
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    
    # Calculate the validation ratio relative to the remaining train_val set
    val_relative_ratio = val_ratio / (1.0 - test_ratio)
    
    # Split train_val into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_ratio, random_state=random_state, stratify=y_train_val
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_sets(X_train, y_train, X_val, y_val, X_test, y_test, path="data/processed"):
    """
    Save the split datasets as .npy files into the designated folder.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    np.save(path / "X_train.npy", X_train)
    np.save(path / "y_train.npy", y_train)
    np.save(path / "X_val.npy", X_val)
    np.save(path / "y_val.npy", y_val)
    np.save(path / "X_test.npy", X_test)
    np.save(path / "y_test.npy", y_test)
    logger.success(f"Successfully saved all split sets to {path}")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

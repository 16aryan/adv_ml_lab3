from pathlib import Path

from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import typer

from lab3.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def print_classifier_scores(y_true, y_pred, set_name="Training"):
    """
    Calculate and display Accuracy and F1 scores.
    """
    acc = accuracy_score(y_true, y_pred)
    # Check if target is binary or multiclass
    try:
        f1 = f1_score(y_true, y_pred, average="binary")
        f1_type = "Binary"
    except ValueError:
        f1 = f1_score(y_true, y_pred, average="macro")
        f1_type = "Macro"
        
    print(f"{set_name} Set Scores:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score ({f1_type}): {f1:.4f}")



@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()

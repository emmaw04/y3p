# src/utils/make_confusion_matrices.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# hard-coded paths
STAGE1_INPUT = Path("runs/final_run/stage1_binary/artifacts/oof_predictions.csv")
STAGE2_INPUT = Path("runs/final_run/stage2_multiclass/artifacts/oof_predictions.csv")
OUTPUT_DIR = Path("runs/confusion_matrix")

# hard-coded settings
STAGE1_THRESHOLD = 0.312  # set this to your chosen threshold
STAGE1_LABELS_NUM = [0, 1]
STAGE1_LABELS_TEXT = ["No pit", "Pit"]

STAGE2_LABELS = ["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"]
STAGE2_PROB_COLS = [f"meta_proba_c{i}" for i in range(len(STAGE2_LABELS))]


def save_confusion_matrix_csv_and_plot(
    cm,
    csv_labels,
    plot_labels,
    csv_path: Path,
    png_path: Path,
    title: str,
    figsize: tuple[float, float] = (7, 6),
) -> None:
    cm_df = pd.DataFrame(cm, index=csv_labels, columns=csv_labels)
    cm_df.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=plot_labels)
    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=True,
        values_format="d",
    )

    # label the colour bar clearly
    if disp.im_ is not None and disp.im_.colorbar is not None:
        disp.im_.colorbar.set_label("Count")

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # stage 1
    df1 = pd.read_csv(STAGE1_INPUT)
    y_true_stage1 = df1["y_pit"]
    y_pred_stage1 = (df1["meta_proba"] >= STAGE1_THRESHOLD).astype(int)

    cm_stage1 = confusion_matrix(y_true_stage1, y_pred_stage1, labels=STAGE1_LABELS_NUM)
    save_confusion_matrix_csv_and_plot(
        cm=cm_stage1,
        csv_labels=STAGE1_LABELS_TEXT,
        plot_labels=STAGE1_LABELS_TEXT,
        csv_path=OUTPUT_DIR / "stage1_meta_confusion_matrix.csv",
        png_path=OUTPUT_DIR / "stage1_meta_confusion_matrix.png",
        title=f"Stage 1 Confusion Matrix (threshold={STAGE1_THRESHOLD:.3f})",
        figsize=(7, 6),
    )

    # stage 2
    df2 = pd.read_csv(STAGE2_INPUT)
    y_true_stage2 = df2["y_compound"]
    y_pred_stage2 = df2[STAGE2_PROB_COLS].idxmax(axis=1).map(
        {f"meta_proba_c{i}": label for i, label in enumerate(STAGE2_LABELS)}
    )

    cm_stage2 = confusion_matrix(y_true_stage2, y_pred_stage2, labels=STAGE2_LABELS)
    save_confusion_matrix_csv_and_plot(
        cm=cm_stage2,
        csv_labels=STAGE2_LABELS,
        plot_labels=STAGE2_LABELS,
        csv_path=OUTPUT_DIR / "stage2_meta_confusion_matrix.csv",
        png_path=OUTPUT_DIR / "stage2_meta_confusion_matrix.png",
        title="Stage 2 Confusion Matrix",
        figsize=(8, 7),
    )

    print(f"Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
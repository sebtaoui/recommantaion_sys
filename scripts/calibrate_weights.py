import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def load_dataset(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as dataset_file:
        return [json.loads(line) for line in dataset_file if line.strip()]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = np.sum(labels == 1)
    negatives = np.sum(labels == 0)

    if positives == 0 or negatives == 0:
        return 0.0

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, len(scores) + 1)

    positive_ranks = ranks[labels == 1]
    sum_positive_ranks = np.sum(positive_ranks)

    auc = (sum_positive_ranks - (positives * (positives + 1) / 2)) / (positives * negatives)
    return float(np.clip(auc, 0.0, 1.0))


def generate_weight_grid(step: float = 0.1) -> Iterable[Tuple[float, float, float, float]]:
    values = np.arange(0, 1 + step, step)
    for weights in itertools.product(values, repeat=4):
        s = sum(weights)
        if s == 0:
            continue
        normalized = tuple(w / s for w in weights)
        yield normalized


def compute_combined_score(sample: Dict, weights: Tuple[float, float, float, float]) -> float:
    embedding_w, cross_w, keywords_w, experience_w = weights
    return (
        embedding_w * sample.get("embedding_component", 0.0)
        + cross_w * sample.get("cross_encoder_component", sample.get("reranker_probability", 0.0))
        + keywords_w * sample.get("keyword_component", 0.0)
        + experience_w * sample.get("experience_component", 0.0)
    )


def calibrate(
    dataset: List[Dict],
    grid_step: float,
) -> Tuple[Tuple[float, float, float, float], float]:
    labels = np.array([sample["label"] for sample in dataset], dtype=np.float32)
    best_auc = -1.0
    best_weights = None

    for weights in generate_weight_grid(step=grid_step):
        scores = np.array(
            [compute_combined_score(sample, weights) for sample in dataset],
            dtype=np.float32,
        )
        auc = roc_auc(scores, labels)

        if auc > best_auc:
            best_auc = auc
            best_weights = weights

    if best_weights is None:
        raise RuntimeError("No valid weights found during calibration.")

    return best_weights, best_auc


def save_profile(
    output_path: Path,
    weights: Tuple[float, float, float, float],
    faiss_preselection_k: int,
    auc_score: float,
) -> None:
    profile = {
        "faiss_preselection_k": faiss_preselection_k,
        "fusion": {
            "embedding": weights[0],
            "cross_encoder": weights[1],
            "keywords": weights[2],
            "experience": weights[3],
        },
        "metadata": {
            "auc": auc_score,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as profile_file:
        json.dump(profile, profile_file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate fusion weights on annotated dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a JSONL file containing annotated samples.",
    )
    parser.add_argument(
        "--output",
        default="calibration/weights.json",
        help="Path to the output calibration profile.",
    )
    parser.add_argument(
        "--faiss-k",
        type=int,
        default=100,
        help="FAISS preselection size to store in the calibration profile.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.1,
        help="Granularity of the grid search (e.g. 0.1, 0.05). Lower = finer but slower.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_dataset(dataset_path)
    if not dataset:
        raise ValueError("Dataset is empty.")

    best_weights, best_auc = calibrate(dataset, args.grid_step)

    output_path = Path(args.output)
    save_profile(output_path, best_weights, args.faiss_k, best_auc)

    print("Calibration completed.")
    print(f"Best weights (embedding, cross, keywords, experience): {best_weights}")
    print(f"ROC-AUC achieved: {best_auc:.4f}")
    print(f"Profile saved to: {output_path}")


if __name__ == "__main__":
    main()



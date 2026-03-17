"""
build_splits_jamendo.py - Build splits.json from MTG-Jamendo metadata.

Run:
    python scripts/build_splits_jamendo.py --metadata path/to/metadata.tsv --label-col genre
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from scripts.config import RANDOM_SEED, SPLITS_FILE, TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from scripts.jamendo_utils import load_metadata, find_column, normalize_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Jamendo metadata TSV/CSV path.")
    parser.add_argument("--label-col", default=None, help="Column containing genre label.")
    parser.add_argument("--id-col", default=None, help="Column containing track id.")
    parser.add_argument("--out", default=str(SPLITS_FILE), help="Output splits.json path.")
    args = parser.parse_args()

    df = load_metadata(Path(args.metadata))
    label_col = args.label_col or find_column(df, ["genre", "genre_top", "tag", "label"])
    ids = normalize_ids(df, args.id_col)
    labels = df[label_col].astype(str)

    # Filter missing labels
    mask = labels.notna()
    ids = ids[mask]
    labels = labels[mask]

    ids_train, ids_tmp, y_train, y_tmp = train_test_split(
        ids, labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=labels,
        random_state=RANDOM_SEED,
    )
    rel_val = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    ids_val, ids_test, y_val, y_test = train_test_split(
        ids_tmp, y_tmp,
        test_size=(1 - rel_val),
        stratify=y_tmp,
        random_state=RANDOM_SEED,
    )

    genres = sorted(set(labels))
    label_map = {g: i for i, g in enumerate(genres)}

    splits = {
        "label_map": label_map,
        "train": {str(tid): label_map[g] for tid, g in zip(ids_train, y_train)},
        "val":   {str(tid): label_map[g] for tid, g in zip(ids_val, y_val)},
        "test":  {str(tid): label_map[g] for tid, g in zip(ids_test, y_test)},
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(splits, indent=2))
    print(f"Wrote splits to {out_path}")


if __name__ == "__main__":
    main()

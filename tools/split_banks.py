import argparse
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_CLASS_NAMES = ["Pedestrian", "Car", "Cyclist"]
BANK_KEYS = ["visual", "mu", "sigma", "mu_log", "V_log", "inv_std_log"]


def load_class_layout(data: np.lib.npyio.NpzFile) -> Tuple[List[str], List[Tuple[int, int]]]:
    class_names = [str(x) for x in data["class_names"].tolist()] if "class_names" in data else list(DEFAULT_CLASS_NAMES)

    if "class_offsets" in data:
        offsets = [int(x) for x in data["class_offsets"].tolist()]
    else:
        counts = [int(x) for x in data["class_counts"].tolist()]
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)

    spans = [(offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]
    return class_names, spans


def sanitize_class_name(name: str) -> str:
    return name.lower().replace(" ", "_")


def split_bank(input_path: str, out_dir: str, class_names_override: Optional[Sequence[str]] = None) -> None:
    data = np.load(input_path)
    class_names, spans = load_class_layout(data)

    if class_names_override is not None:
        class_names = list(class_names_override)

    os.makedirs(out_dir, exist_ok=True)

    for class_name, (start, end) in zip(class_names, spans):
        bank = {k: data[k][start:end] for k in BANK_KEYS}
        if "counts" in data:
            bank["counts"] = data["counts"][start:end]

        out_name = f"priors_{sanitize_class_name(class_name)}.npz"
        out_path = os.path.join(out_dir, out_name)
        np.savez(out_path, **bank)
        print(f"[saved] {out_path} ({end - start} prototypes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="split a prior bank into per-class .npz files")
    parser.add_argument("--input", type=str, default="priors/priors_unified.npz", help="path to bank")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="output directory for class banks (default: same directory as --input)",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="optional override for output class names/order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input
    if not os.path.isabs(input_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        input_path = os.path.join(repo_root, input_path)

    if args.out_dir is None:
        out_dir = os.path.dirname(input_path)
    else:
        out_dir = args.out_dir
        if not os.path.isabs(out_dir):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            out_dir = os.path.join(repo_root, out_dir)

    print(f"[input] {input_path}")
    print(f"[out]   {out_dir}")
    split_bank(input_path=input_path, out_dir=out_dir, class_names_override=args.class_names)


if __name__ == "__main__":
    main()

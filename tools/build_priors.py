import argparse
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm


CLIP_MODEL_NAME = "RN50"


def require_clip():
    try:
        import clip  # type: ignore
    except ImportError as e:
        raise ImportError(
            "CLIP is required for prior generation. Install with: "
            "pip install git+https://github.com/openai/CLIP.git"
        ) from e
    return clip


@dataclass
class ClassPriorCfg:
    k_geo: int
    k_vis: int
    min_bbox_height: int
    max_occlusion: int
    max_truncation: float
    max_inv_std: float
    min_eig: float
    dedup_cos: float
    min_cluster_size: int
    min_sigma: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic mode for release reproducibility of mined priors.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_cfg(cfg_path: str) -> Dict:
    with open(cfg_path, "r") as f:
        return yaml.load(f, Loader=yaml.Loader)


def parse_kitti_label(label_file: str) -> List[Dict]:
    objects = []
    with open(label_file, "r") as f:
        for line in f:
            data = line.strip().split(" ")
            if len(data) < 15:
                continue
            objects.append(
                {
                    "type": data[0],
                    "truncation": float(data[1]),
                    "occlusion": int(data[2]),
                    "bbox": [float(x) for x in data[4:8]],
                    "dims": [float(x) for x in data[8:11]],  # h,w,l
                }
            )
    return objects


def pad_to_square(img_crop: np.ndarray) -> np.ndarray:
    h, w = img_crop.shape[:2]
    target = max(h, w)
    top = (target - h) // 2
    bottom = target - h - top
    left = (target - w) // 2
    right = target - w - left
    return cv2.copyMakeBorder(
        img_crop,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def compute_whitening(feats: np.ndarray) -> np.ndarray:
    # whitening is only used for clustering stability; stored visual centroids stay in original clip space.
    n_components = min(feats.shape[0], feats.shape[1])
    pca = PCA(n_components=n_components, whiten=True, svd_solver="auto", random_state=42)
    return pca.fit_transform(feats)


def merge_unified(out_dir: str) -> str:
    car = np.load(os.path.join(out_dir, "priors_car.npz"))
    ped = np.load(os.path.join(out_dir, "priors_pedestrian.npz"))
    cyc = np.load(os.path.join(out_dir, "priors_cyclist.npz"))

    # class order contract with router/model class ids: [Pedestrian, Car, Cyclist].
    banks = [ped, car, cyc]
    visuals, mu, sigma, mu_log, V_log, inv_std_log = [], [], [], [], [], []
    offsets = [0]
    class_counts = []
    for b in banks:
        count = int(b["mu"].shape[0])
        offsets.append(offsets[-1] + count)
        class_counts.append(count)
        visuals.append(b["visual"])
        mu.append(b["mu"])
        sigma.append(b["sigma"])
        mu_log.append(b["mu_log"])
        V_log.append(b["V_log"])
        inv_std_log.append(b["inv_std_log"])

    unified_path = os.path.join(out_dir, "priors_unified.npz")
    np.savez(
        unified_path,
        visual=np.concatenate(visuals, axis=0),
        mu=np.concatenate(mu, axis=0),
        sigma=np.concatenate(sigma, axis=0),
        mu_log=np.concatenate(mu_log, axis=0),
        V_log=np.concatenate(V_log, axis=0),
        inv_std_log=np.concatenate(inv_std_log, axis=0),
        class_offsets=np.array(offsets, dtype=np.int64),
        class_counts=np.array(class_counts, dtype=np.int64),
        class_names=np.array(["Pedestrian", "Car", "Cyclist"]),
    )
    return unified_path


def build_single_class_bank(
    *,
    root_dir: str,
    class_name: str,
    class_cfg: ClassPriorCfg,
    image_ids: List[str],
    device: str,
    output_path: str,
) -> None:
    image_dir = os.path.join(root_dir, "training", "image_2")
    label_dir = os.path.join(root_dir, "training", "label_2")

    valid_dims: List[np.ndarray] = []
    valid_embeddings: List[np.ndarray] = []

    clip = require_clip()
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    model.eval()

    for image_id in tqdm(image_ids, desc=f"collect:{class_name}"):
        label_file = os.path.join(label_dir, f"{image_id}.txt")
        if not os.path.exists(label_file):
            continue
        img_path = os.path.join(image_dir, f"{image_id}.png")
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        objects = parse_kitti_label(label_file)
        for obj in objects:
            if obj["type"] != class_name:
                continue
            bbox = obj["bbox"]
            if (bbox[3] - bbox[1]) < class_cfg.min_bbox_height:
                continue
            if obj["occlusion"] > class_cfg.max_occlusion:
                continue
            if obj["truncation"] > class_cfg.max_truncation:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            padded_crop = pad_to_square(crop)
            pil_img = Image.fromarray(padded_crop)
            clip_input = preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = model.encode_image(clip_input)
                feat = feat / feat.norm(dim=-1, keepdim=True)

            dims = np.array([obj["dims"][0], obj["dims"][1], obj["dims"][2]], dtype=np.float32)
            valid_dims.append(dims)
            valid_embeddings.append(feat.cpu().numpy())

    if len(valid_dims) == 0:
        raise ValueError(f"No valid samples for class: {class_name}")

    valid_dims_np = np.stack(valid_dims, axis=0)
    valid_embeddings_np = np.vstack(valid_embeddings)
    valid_embeddings_np = valid_embeddings_np / np.linalg.norm(valid_embeddings_np, axis=1, keepdims=True)

    feats_for_cluster = compute_whitening(valid_embeddings_np)

    # geometry stage in log-size space (matches multiplicative size variation).
    dims_log = np.log(valid_dims_np + 1e-6)
    mu_global = np.mean(dims_log, axis=0)
    std_global = np.std(dims_log, axis=0)
    std_global = np.clip(std_global, 1e-8, None)
    dims_norm = (dims_log - mu_global) / std_global

    kmeans_geo = KMeans(n_clusters=class_cfg.k_geo, random_state=42, n_init=10)
    geo_labels = kmeans_geo.fit_predict(dims_norm)

    final_labels = np.zeros_like(geo_labels)
    total_prototypes = 0
    for geo_id in range(class_cfg.k_geo):
        # two-stage clustering: geometry groups first, then visual sub-groups inside each geometry group.
        indices = np.where(geo_labels == geo_id)[0]
        if len(indices) < class_cfg.k_vis * 3:
            # if support is too small, keep a single prototype for this geometry group.
            final_labels[indices] = total_prototypes
            total_prototypes += 1
            continue
        sub_feats = feats_for_cluster[indices]
        kmeans_vis = KMeans(n_clusters=class_cfg.k_vis, random_state=42, n_init=10)
        vis_labels = kmeans_vis.fit_predict(sub_feats)
        final_labels[indices] = total_prototypes + vis_labels
        total_prototypes += class_cfg.k_vis

    cluster_members = []
    centroids_raw = []
    for k in range(total_prototypes):
        members = np.where(final_labels == k)[0]
        if members.size == 0:
            cluster_members.append(None)
            centroids_raw.append(None)
            continue
        cluster_members.append(members)
        cent = valid_embeddings_np[members].mean(axis=0)
        cent = cent / np.linalg.norm(cent)
        centroids_raw.append(cent)

    parent = list(range(total_prototypes))
    # merge near-duplicate visual prototypes using class-specific cosine threshold.
    for i in range(total_prototypes):
        if centroids_raw[i] is None:
            continue
        for j in range(i + 1, total_prototypes):
            if centroids_raw[j] is None:
                continue
            cos = float(np.dot(centroids_raw[i], centroids_raw[j]))
            if cos >= class_cfg.dedup_cos:
                parent[j] = parent[i]

    for i in range(total_prototypes):
        root = i
        while parent[root] != root:
            root = parent[root]
        parent[i] = root

    unique_parents = sorted(set(parent))
    remap = {p: idx for idx, p in enumerate(unique_parents)}
    merged = {remap[p]: [] for p in unique_parents}
    for idx, p in enumerate(parent):
        if centroids_raw[idx] is None:
            continue
        merged[remap[p]].extend(cluster_members[idx].tolist())

    bank_mu_linear = []
    bank_sigma_linear = []
    bank_mu_log = []
    bank_V_log = []
    bank_inv_std_log = []
    bank_visual = []
    cluster_counts = []

    for _, member_idx in merged.items():
        member_idx = np.asarray(member_idx, dtype=np.int64)
        cluster_dims = np.atleast_2d(valid_dims_np[member_idx])
        cluster_feats = valid_embeddings_np[member_idx]
        count = int(cluster_dims.shape[0])

        mu_lin = np.mean(cluster_dims, axis=0)
        sigma_lin = np.std(cluster_dims, axis=0)
        if class_cfg.min_sigma > 0:
            sigma_lin = np.maximum(sigma_lin, class_cfg.min_sigma)

        cluster_log_dims = np.log(cluster_dims + 1e-6)
        mu_log = np.mean(cluster_log_dims, axis=0)

        # stabilize manifold statistics for low-support clusters before svd/inv-std computation.
        cov_log = np.cov(cluster_log_dims, rowvar=False)
        if np.ndim(cov_log) == 0:
            cov_log = np.eye(3) * float(cov_log)
        if count < 3:
            cov_log = np.eye(3) * class_cfg.min_eig
        cov_log = np.nan_to_num(cov_log, nan=class_cfg.min_eig, posinf=class_cfg.min_eig, neginf=class_cfg.min_eig)
        cov_log += np.eye(3) * 1e-5
        if count < 50:
            rho = 0.1
            trace = np.trace(cov_log)
            cov_log = (1 - rho) * cov_log + rho * np.eye(3) * (trace / 3.0)

        _, S, Vt = np.linalg.svd(cov_log)
        S = np.maximum(S, class_cfg.min_eig)
        inv_std = 1.0 / np.sqrt(S)
        inv_std = np.minimum(inv_std, class_cfg.max_inv_std)
        V_matrix = Vt.T

        vis_centroid = np.mean(cluster_feats, axis=0)
        vis_centroid = vis_centroid / np.linalg.norm(vis_centroid)

        bank_mu_linear.append(mu_lin)
        bank_sigma_linear.append(sigma_lin)
        bank_mu_log.append(mu_log)
        bank_V_log.append(V_matrix)
        bank_inv_std_log.append(inv_std)
        bank_visual.append(vis_centroid)
        cluster_counts.append(count)

    if class_cfg.min_cluster_size > 1 and len(cluster_counts) > 1:
        # merge tiny clusters into nearest visual centroid to avoid brittle low-support prototypes.
        final_mu = np.array(bank_mu_linear)
        final_visual = np.array(bank_visual)
        final_counts = np.array(cluster_counts)
        small_idx = np.where(final_counts < class_cfg.min_cluster_size)[0].tolist()
        if small_idx:
            keep_mask = np.ones(len(final_counts), dtype=bool)
            keep_mask[small_idx] = False
            if np.any(keep_mask):
                keep_visual = final_visual[keep_mask]
                keep_mu = final_mu[keep_mask]
                keep_sigma = np.array(bank_sigma_linear)[keep_mask]
                keep_mu_log = np.array(bank_mu_log)[keep_mask]
                keep_V_log = np.array(bank_V_log)[keep_mask]
                keep_inv_std = np.array(bank_inv_std_log)[keep_mask]
                keep_counts = final_counts[keep_mask]

                for idx in small_idx:
                    vis = final_visual[idx]
                    cos = keep_visual @ vis
                    target = int(np.argmax(cos))
                    w_old = keep_counts[target]
                    w_new = final_counts[idx]
                    total = w_old + w_new
                    keep_counts[target] = total
                    keep_mu[target] = (keep_mu[target] * w_old + final_mu[idx] * w_new) / total
                    keep_sigma[target] = (keep_sigma[target] * w_old + np.array(bank_sigma_linear)[idx] * w_new) / total
                    keep_mu_log[target] = (keep_mu_log[target] * w_old + np.array(bank_mu_log)[idx] * w_new) / total
                    # keep conservative uncertainty after merge.
                    keep_inv_std[target] = np.minimum(keep_inv_std[target], np.array(bank_inv_std_log)[idx])

                bank_mu_linear = keep_mu.tolist()
                bank_sigma_linear = keep_sigma.tolist()
                bank_mu_log = keep_mu_log.tolist()
                bank_V_log = keep_V_log.tolist()
                bank_inv_std_log = keep_inv_std.tolist()
                bank_visual = keep_visual.tolist()
                cluster_counts = keep_counts.tolist()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        visual=np.array(bank_visual),
        mu=np.array(bank_mu_linear),
        sigma=np.array(bank_sigma_linear),
        mu_log=np.array(bank_mu_log),
        V_log=np.array(bank_V_log),
        inv_std_log=np.array(bank_inv_std_log),
        counts=np.array(cluster_counts, dtype=np.int64),
    )


def default_class_cfgs() -> Dict[str, ClassPriorCfg]:
    # default recipe used for final paper priors.
    return {
        "Car": ClassPriorCfg(
            k_geo=5,
            k_vis=4,
            min_bbox_height=30,
            max_occlusion=0,
            max_truncation=0.0,
            max_inv_std=10.0,
            min_eig=1e-3,
            dedup_cos=0.999,
            min_cluster_size=10,
            min_sigma=0.03,
        ),
        "Pedestrian": ClassPriorCfg(
            k_geo=4,
            k_vis=2,
            min_bbox_height=15,
            max_occlusion=2,
            max_truncation=0.3,
            max_inv_std=6.0,
            min_eig=0.002,
            dedup_cos=0.9995,
            min_cluster_size=30,
            min_sigma=0.05,
        ),
        "Cyclist": ClassPriorCfg(
            k_geo=4,
            k_vis=2,
            min_bbox_height=15,
            max_occlusion=2,
            max_truncation=0.3,
            max_inv_std=8.0,
            min_eig=0.002,
            dedup_cos=0.9995,
            min_cluster_size=12,
            min_sigma=0.05,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="build MonoPRIO prior banks from KITTI labels/images")
    parser.add_argument("--config", type=str, default="configs/monoprio.yaml")
    parser.add_argument("--split", type=str, default=None, help="override dataset.train_split from config")
    parser.add_argument("--out-dir", type=str, default=None, help="output directory for class banks and unified prior")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-copy-default", action="store_true", help="do not copy train prior to priors/priors_unified.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(repo_root, cfg_path)
    cfg = load_cfg(cfg_path)

    root_dir = cfg["dataset"]["root_dir"]
    if not os.path.isabs(root_dir):
        root_dir = os.path.abspath(os.path.join(repo_root, root_dir))
    split_name = args.split if args.split is not None else cfg["dataset"].get("train_split", "train")
    split_file = os.path.join(root_dir, "ImageSets", f"{split_name}.txt")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"missing split file: {split_file}")

    with open(split_file, "r") as f:
        image_ids = [line.strip() for line in f if line.strip()]
    if len(image_ids) == 0:
        raise ValueError(f"split has no ids: {split_file}")

    if args.out_dir is None:
        out_dir = os.path.join(repo_root, "priors", split_name)
    else:
        out_dir = args.out_dir
        if not os.path.isabs(out_dir):
            out_dir = os.path.abspath(os.path.join(repo_root, out_dir))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[cfg] {cfg_path}")
    print(f"[root] {root_dir}")
    print(f"[split] {split_name} ({len(image_ids)} ids)")
    print(f"[out] {out_dir}")
    print(f"[device] {args.device}")

    class_cfgs = default_class_cfgs()
    for class_name in ["Car", "Pedestrian", "Cyclist"]:
        output_path = os.path.join(out_dir, f"priors_{class_name.lower()}.npz")
        build_single_class_bank(
            root_dir=root_dir,
            class_name=class_name,
            class_cfg=class_cfgs[class_name],
            image_ids=image_ids,
            device=args.device,
            output_path=output_path,
        )
        print(f"[saved] {output_path}")

    unified_path = merge_unified(out_dir)
    print(f"[saved] {unified_path}")

    copy_default = (not args.no_copy_default) and (split_name == "train")
    if copy_default:
        # keep the default train prior path expected by configs/monoprio.yaml.
        dst = os.path.join(repo_root, "priors", "priors_unified.npz")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(unified_path, dst)
        print(f"[saved] {dst}")


if __name__ == "__main__":
    main()

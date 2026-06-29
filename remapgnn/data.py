from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


INDEX_SRC_COL = "source_index"
INDEX_TGT_COL = "target_index"


@dataclass(frozen=True)
class PairInfo:
    pair: str
    src: str
    tgt: str
    src_mesh: str
    tgt_mesh: str


def split_pair(pair: str) -> PairInfo:
    src, tgt = pair.split("_to_")
    return PairInfo(
        pair=pair,
        src=src,
        tgt=tgt,
        src_mesh=src.split("-")[0],
        tgt_mesh=tgt.split("-")[0],
    )


def edge_path(cfg, pair: str) -> Path:
    return cfg.edge_path(pair)


def map_path(cfg, pair: str) -> Path:
    return cfg.map_path(pair)


def field_paths(cfg, pair: str) -> tuple[Path, Path]:
    return cfg.source_target_files(pair)



def _mesh_family_name(x) -> str:
    x = str(x).upper()
    if "RLL" in x:
        return "RLL"
    if "ICOD" in x:
        return "ICOD"
    if "CS" in x:
        return "CS"
    return "OTHER"


def add_mesh_condition_columns(df):
    """
    Add numeric mesh-family conditioning columns.

    These are constant across each pair but useful as global conditioning signals:
      src_mesh_is_RLL, src_mesh_is_CS, src_mesh_is_ICOD
      tgt_mesh_is_RLL, tgt_mesh_is_CS, tgt_mesh_is_ICOD
    """
    df = df.copy()

    if "src_mesh" in df.columns:
        src_family = df["src_mesh"].map(_mesh_family_name)
    elif "pair" in df.columns:
        src_family = df["pair"].astype(str).str.split("_to_").str[0].map(_mesh_family_name)
    else:
        src_family = None

    if "tgt_mesh" in df.columns:
        tgt_family = df["tgt_mesh"].map(_mesh_family_name)
    elif "pair" in df.columns:
        tgt_family = df["pair"].astype(str).str.split("_to_").str[1].map(_mesh_family_name)
    else:
        tgt_family = None

    for fam in ["RLL", "CS", "ICOD"]:
        if src_family is not None:
            df[f"src_mesh_is_{fam}"] = (src_family == fam).astype("float32")
        if tgt_family is not None:
            df[f"tgt_mesh_is_{fam}"] = (tgt_family == fam).astype("float32")

    df = add_geometry_feature_columns(df)
    return df


SYNTHETIC_GEOMETRY_FEATURES = {
    "src_h",
    "tgt_h",
    "log_src_area",
    "log_tgt_area",
    "log_area_ratio_tgt_over_src",
    "target_candidate_count",
    "source_candidate_count",
    "target_candidate_count_log",
    "source_candidate_count_log",
    "knn_rank_over_target_count",
    "tgt_tan_e",
    "tgt_tan_n",
    "src_tan_e",
    "src_tan_n",
    "tgt_tan_e_over_h_tgt",
    "tgt_tan_n_over_h_tgt",
    "tgt_tan_e_over_h_mean",
    "tgt_tan_n_over_h_mean",
    "src_tan_e_over_h_src",
    "src_tan_n_over_h_src",
    "tan_dist",
    "tan_dist_over_h_src",
    "tan_dist_over_h_tgt",
    "tan_dist_over_h_mean",
    "tgt_tan_e2_over_h2",
    "tgt_tan_en_over_h2",
    "tgt_tan_n2_over_h2",
}

GEOMETRY_FEATURE_DEPENDENCIES = [
    "source_index",
    "target_index",
    "knn_rank",
    "src_x",
    "src_y",
    "src_z",
    "tgt_x",
    "tgt_y",
    "tgt_z",
    "src_area",
    "tgt_area",
    "area_ratio_tgt_over_src",
]

MESH_CONDITION_FEATURES = {
    "src_mesh_is_RLL", "src_mesh_is_CS", "src_mesh_is_ICOD",
    "tgt_mesh_is_RLL", "tgt_mesh_is_CS", "tgt_mesh_is_ICOD",
}


def _stable_tangent_basis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable east/north-like tangent basis for each unit-sphere point."""
    p = np.asarray(xyz, dtype=np.float64)
    p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1.0e-30)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    east = np.cross(z_axis[None, :], p)
    bad = np.linalg.norm(east, axis=1) < 1.0e-10
    if np.any(bad):
        east[bad] = np.cross(x_axis[None, :], p[bad])
    east = east / np.maximum(np.linalg.norm(east, axis=1, keepdims=True), 1.0e-30)

    north = np.cross(p, east)
    north = north / np.maximum(np.linalg.norm(north, axis=1, keepdims=True), 1.0e-30)
    return east, north


def add_geometry_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add deployable local-geometry features derived from existing edge columns.

    These are intentionally supermesh-free: only cell centers, areas, and the
    candidate graph are used. If the required physical columns are absent, this
    is a no-op.
    """
    required = {
        "source_index",
        "target_index",
        "src_x",
        "src_y",
        "src_z",
        "tgt_x",
        "tgt_y",
        "tgt_z",
        "src_area",
        "tgt_area",
    }
    if not required.issubset(df.columns):
        return df

    eps = 1.0e-30
    src_area = df["src_area"].to_numpy(dtype=np.float64, copy=False)
    tgt_area = df["tgt_area"].to_numpy(dtype=np.float64, copy=False)
    src_h = np.sqrt(np.maximum(src_area, eps))
    tgt_h = np.sqrt(np.maximum(tgt_area, eps))
    h_mean = np.sqrt(0.5 * np.maximum(src_area + tgt_area, eps))

    df["src_h"] = src_h.astype("float32")
    df["tgt_h"] = tgt_h.astype("float32")
    df["log_src_area"] = np.log(np.maximum(src_area, eps)).astype("float32")
    df["log_tgt_area"] = np.log(np.maximum(tgt_area, eps)).astype("float32")
    if "area_ratio_tgt_over_src" in df.columns:
        ratio = df["area_ratio_tgt_over_src"].to_numpy(dtype=np.float64, copy=False)
    else:
        ratio = tgt_area / np.maximum(src_area, eps)
    df["log_area_ratio_tgt_over_src"] = np.log(np.maximum(ratio, eps)).astype("float32")

    src_xyz = df[["src_x", "src_y", "src_z"]].to_numpy(dtype=np.float64, copy=False)
    tgt_xyz = df[["tgt_x", "tgt_y", "tgt_z"]].to_numpy(dtype=np.float64, copy=False)
    d_tgt_to_src = src_xyz - tgt_xyz
    d_src_to_tgt = tgt_xyz - src_xyz

    tgt_east, tgt_north = _stable_tangent_basis(tgt_xyz)
    src_east, src_north = _stable_tangent_basis(src_xyz)

    tgt_u = np.sum(d_tgt_to_src * tgt_east, axis=1)
    tgt_v = np.sum(d_tgt_to_src * tgt_north, axis=1)
    src_u = np.sum(d_src_to_tgt * src_east, axis=1)
    src_v = np.sum(d_src_to_tgt * src_north, axis=1)
    tan_dist = np.sqrt(tgt_u * tgt_u + tgt_v * tgt_v)

    df["tgt_tan_e"] = tgt_u.astype("float32")
    df["tgt_tan_n"] = tgt_v.astype("float32")
    df["src_tan_e"] = src_u.astype("float32")
    df["src_tan_n"] = src_v.astype("float32")
    df["tgt_tan_e_over_h_tgt"] = (tgt_u / np.maximum(tgt_h, eps)).astype("float32")
    df["tgt_tan_n_over_h_tgt"] = (tgt_v / np.maximum(tgt_h, eps)).astype("float32")
    df["tgt_tan_e_over_h_mean"] = (tgt_u / np.maximum(h_mean, eps)).astype("float32")
    df["tgt_tan_n_over_h_mean"] = (tgt_v / np.maximum(h_mean, eps)).astype("float32")
    df["src_tan_e_over_h_src"] = (src_u / np.maximum(src_h, eps)).astype("float32")
    df["src_tan_n_over_h_src"] = (src_v / np.maximum(src_h, eps)).astype("float32")
    df["tan_dist"] = tan_dist.astype("float32")
    df["tan_dist_over_h_src"] = (tan_dist / np.maximum(src_h, eps)).astype("float32")
    df["tan_dist_over_h_tgt"] = (tan_dist / np.maximum(tgt_h, eps)).astype("float32")
    df["tan_dist_over_h_mean"] = (tan_dist / np.maximum(h_mean, eps)).astype("float32")

    h2 = np.maximum(h_mean * h_mean, eps)
    df["tgt_tan_e2_over_h2"] = ((tgt_u * tgt_u) / h2).astype("float32")
    df["tgt_tan_en_over_h2"] = ((tgt_u * tgt_v) / h2).astype("float32")
    df["tgt_tan_n2_over_h2"] = ((tgt_v * tgt_v) / h2).astype("float32")

    tgt_count = df.groupby("target_index", sort=False)["source_index"].transform("size").to_numpy(dtype=np.float64)
    src_count = df.groupby("source_index", sort=False)["target_index"].transform("size").to_numpy(dtype=np.float64)
    df["target_candidate_count"] = tgt_count.astype("float32")
    df["source_candidate_count"] = src_count.astype("float32")
    df["target_candidate_count_log"] = np.log1p(tgt_count).astype("float32")
    df["source_candidate_count_log"] = np.log1p(src_count).astype("float32")
    if "knn_rank" in df.columns:
        rank = df["knn_rank"].to_numpy(dtype=np.float64, copy=False)
        df["knn_rank_over_target_count"] = (rank / np.maximum(tgt_count - 1.0, 1.0)).astype("float32")

    return df


def physical_columns_for_features(columns: Iterable[str]) -> list[str]:
    """Columns to read from parquet in order to materialize requested features."""
    requested = set(columns)
    out = []
    for col in requested:
        if col in MESH_CONDITION_FEATURES or col in SYNTHETIC_GEOMETRY_FEATURES:
            continue
        out.append(col)
    if requested & SYNTHETIC_GEOMETRY_FEATURES:
        out.extend(GEOMETRY_FEATURE_DEPENDENCIES)
    out.extend(["src_mesh", "tgt_mesh", "pair"])
    return list(dict.fromkeys(out))


def load_edge_dataframe(path, columns=None):
    if columns is not None:
        columns = physical_columns_for_features(columns)
    df = pd.read_parquet(path, columns=columns)
    return add_mesh_condition_columns(df)

def edge_schema(cfg, pair: str) -> pd.DataFrame:
    df = load_edge_dataframe(cfg.edge_path(pair))
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "n_unique": int(df[col].nunique(dropna=True)) if len(df) <= 500000 else None,
            }
        )
    return pd.DataFrame(rows)


def validate_edge_dataframe(df: pd.DataFrame) -> None:
    missing = [c for c in [INDEX_SRC_COL, INDEX_TGT_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"missing required edge columns: {missing}")


def edge_indices(
    df: pd.DataFrame,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_edge_dataframe(df)
    src_index = torch.as_tensor(df[INDEX_SRC_COL].to_numpy().copy(), dtype=torch.long, device=device)
    tgt_index = torch.as_tensor(df[INDEX_TGT_COL].to_numpy().copy(), dtype=torch.long, device=device)
    return src_index, tgt_index


def infer_node_counts(df: pd.DataFrame) -> tuple[int, int]:
    validate_edge_dataframe(df)
    n_src = int(df[INDEX_SRC_COL].max()) + 1
    n_tgt = int(df[INDEX_TGT_COL].max()) + 1
    return n_src, n_tgt


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def likely_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Conservative feature-column guess.

    This is intentionally broad for inspection. Later, train_config.py
    should use explicit feature lists from the config or the existing
    training script's conventions.
    """
    exclude_exact = {
        INDEX_SRC_COL,
        INDEX_TGT_COL,
        "edge_exists",
        "exists",
        "S_true",
        "M_true",
        "weight",
        "true_weight",
        "area_src",
        "area_tgt",
        "src_area",
        "tgt_area",
    }

    cols = []
    for c in numeric_columns(df):
        if c in exclude_exact:
            continue
        if c.startswith("area_"):
            continue
        cols.append(c)
    return cols


def tensor_from_columns(
    df: pd.DataFrame,
    columns: list[str],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not columns:
        raise ValueError("no columns provided")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    return torch.as_tensor(df[columns].to_numpy().copy(), dtype=dtype, device=device)


def print_pair_data_summary(cfg, pair: str) -> None:
    info = split_pair(pair)
    epath = cfg.edge_path(pair)
    mpath = cfg.map_path(pair)
    src_file, tgt_file = cfg.source_target_files(pair)

    print("=" * 100)
    print(f"pair:        {pair}")
    print(f"src:         {info.src} ({info.src_mesh})")
    print(f"tgt:         {info.tgt} ({info.tgt_mesh})")
    print(f"edge path:   {epath}")
    print(f"map path:    {mpath}")
    print(f"src fields:  {src_file}")
    print(f"tgt fields:  {tgt_file}")
    print()

    df = load_edge_dataframe(cfg.edge_path(pair), columns=[INDEX_SRC_COL, INDEX_TGT_COL])
    n_src, n_tgt = infer_node_counts(df)

    print(f"edges:       {len(df):,}")
    print(f"n_src infer: {n_src:,}")
    print(f"n_tgt infer: {n_tgt:,}")
    print(f"edge exists: {epath.exists()}")
    print(f"map exists:  {mpath.exists()}")
    print(f"src exists:  {src_file.exists()}")
    print(f"tgt exists:  {tgt_file.exists()}")

# ---------------------------------------------------------------------
# Training/evaluation tensor construction
# ---------------------------------------------------------------------

DEFAULT_EDGE_FEATURES = [
    "dx", "dy", "dz",
    "chord_dist",
    "area_ratio_tgt_over_src",
    "knn_rank",
]

DEFAULT_SRC_NODE_FEATURES = [
    "src_x", "src_y", "src_z",
    "src_area",
]

DEFAULT_TGT_NODE_FEATURES = [
    "tgt_x", "tgt_y", "tgt_z",
    "tgt_area",
]


def get_feature_lists(cfg) -> tuple[list[str], list[str], list[str]]:
    """
    Return edge/source-node/target-node features.

    Defaults match the current v10 training script. Config can override later.
    """
    features = cfg.raw.get("features", {})
    edge_features = list(features.get("edge", DEFAULT_EDGE_FEATURES))
    src_node_features = list(features.get("src_node", DEFAULT_SRC_NODE_FEATURES))
    tgt_node_features = list(features.get("tgt_node", DEFAULT_TGT_NODE_FEATURES))
    return edge_features, src_node_features, tgt_node_features


def compute_feature_stats(
    cfg,
    pairs: list[str],
    sample_per_pair: int = 80000,
    seed: int = 123,
) -> dict:
    """
    Compute global normalization stats exactly like the current training scripts.
    """
    import numpy as np

    edge_features, src_node_features, tgt_node_features = get_feature_lists(cfg)

    print("Computing global feature normalization stats from samples...")

    edge_chunks = []
    src_chunks = []
    tgt_chunks = []

    for pair in pairs:
        p = cfg.edge_path(pair)
        print(f"  reading sample: {p}")

        cols = list(set(edge_features + src_node_features + tgt_node_features))
        # Some configured features are synthetic and not stored directly in
        # parquet. Read their physical dependencies, then materialize them via
        # add_mesh_condition_columns/add_geometry_feature_columns below.
        physical_cols = physical_columns_for_features(cols)

        df = pd.read_parquet(p, columns=physical_cols)
        df = add_mesh_condition_columns(df)

        if len(df) > sample_per_pair:
            df = df.sample(sample_per_pair, random_state=seed)

        edge_chunks.append(df[edge_features].to_numpy(dtype="float32").copy())
        src_chunks.append(df[src_node_features].to_numpy(dtype="float32").copy())
        tgt_chunks.append(df[tgt_node_features].to_numpy(dtype="float32").copy())

    edge_X = np.concatenate(edge_chunks, axis=0)
    src_X = np.concatenate(src_chunks, axis=0)
    tgt_X = np.concatenate(tgt_chunks, axis=0)

    stats = {
        "edge_mean": edge_X.mean(axis=0, keepdims=True).astype("float32"),
        "edge_std": (edge_X.std(axis=0, keepdims=True) + 1.0e-12).astype("float32"),
        "src_mean": src_X.mean(axis=0, keepdims=True).astype("float32"),
        "src_std": (src_X.std(axis=0, keepdims=True) + 1.0e-12).astype("float32"),
        "tgt_mean": tgt_X.mean(axis=0, keepdims=True).astype("float32"),
        "tgt_std": (tgt_X.std(axis=0, keepdims=True) + 1.0e-12).astype("float32"),
    }

    print("Feature stats ready.")
    return stats


def unique_node_features(
    df: pd.DataFrame,
    index_col: str,
    feature_cols: list[str],
    n_nodes: int,
):
    """
    Build one feature row per node from an edge dataframe.

    This matches the current training script behavior:
      out[idx] = vals
    """
    import numpy as np

    out = np.zeros((n_nodes, len(feature_cols)), dtype=np.float32)
    idx = df[index_col].to_numpy(dtype="int64").copy()
    vals = df[feature_cols].to_numpy(dtype="float32").copy()
    out[idx] = vals
    return out


def load_pair_tensors(
    cfg,
    pair: str,
    stats: dict,
    device: torch.device | str,
) -> dict:
    """
    Load one mesh-pair edge dataset and return the training/eval tensors.

    This mirrors load_pair_tensors() from the v10 training script.
    """
    import numpy as np

    p = cfg.edge_path(pair)
    return load_pair_tensors_from_path(
        p,
        cfg,
        stats,
        device=device,
        pair=pair,
    )


def load_pair_tensors_from_path(
    edge_path: str | Path,
    cfg,
    stats: dict,
    device: torch.device | str,
    *,
    pair: str | None = None,
    feature_lists: tuple[list[str], list[str], list[str]] | None = None,
) -> dict:
    """
    Load one prepared source-target edge dataset from an explicit parquet path.

    This is the inference/deployment companion to ``load_pair_tensors``.  It is
    useful for externally prepared meshes whose edge parquet is not located at
    ``cfg.edge_path(pair)``.

    ``feature_lists`` defaults to the config's feature lists, but inference
    should usually pass the feature lists stored in the checkpoint pack so the
    tensor layout exactly matches training.
    """
    import numpy as np

    if feature_lists is None:
        edge_features, src_node_features, tgt_node_features = get_feature_lists(cfg)
    else:
        edge_features, src_node_features, tgt_node_features = feature_lists

    df = pd.read_parquet(edge_path)
    if pair is not None and "pair" not in df.columns:
        df = df.copy()
        df["pair"] = pair
    df = add_mesh_condition_columns(df)

    src_index_np = df["source_index"].to_numpy(dtype="int64").copy()
    tgt_index_np = df["target_index"].to_numpy(dtype="int64").copy()

    n_src = int(src_index_np.max()) + 1
    n_tgt = int(tgt_index_np.max()) + 1

    edge_np = df[edge_features].to_numpy(dtype="float32").copy()
    edge_np = (edge_np - stats["edge_mean"]) / stats["edge_std"]

    src_node_np = unique_node_features(df, "source_index", src_node_features, n_src)
    tgt_node_np = unique_node_features(df, "target_index", tgt_node_features, n_tgt)

    src_node_np = (src_node_np - stats["src_mean"]) / stats["src_std"]
    tgt_node_np = (tgt_node_np - stats["tgt_mean"]) / stats["tgt_std"]

    if "edge_exists" in df.columns:
        edge_exists_np = df["edge_exists"].to_numpy(dtype="float32").copy()
    else:
        edge_exists_np = np.zeros(len(df), dtype=np.float32)
    if "weight" in df.columns:
        S_true_np = df["weight"].to_numpy(dtype="float32").copy()
    else:
        S_true_np = np.zeros(len(df), dtype=np.float32)

    area_src_np = np.zeros(n_src, dtype=np.float32)
    area_tgt_np = np.zeros(n_tgt, dtype=np.float32)
    area_src_np[src_index_np] = df["src_area"].to_numpy(dtype="float32").copy()
    area_tgt_np[tgt_index_np] = df["tgt_area"].to_numpy(dtype="float32").copy()

    return {
        "pair": pair,
        "edge_attr": torch.tensor(edge_np, dtype=torch.float32, device=device),
        "src_node_attr": torch.tensor(src_node_np, dtype=torch.float32, device=device),
        "tgt_node_attr": torch.tensor(tgt_node_np, dtype=torch.float32, device=device),
        "src_index": torch.tensor(src_index_np, dtype=torch.long, device=device),
        "tgt_index": torch.tensor(tgt_index_np, dtype=torch.long, device=device),
        "edge_exists": torch.tensor(edge_exists_np, dtype=torch.float32, device=device),
        "S_true": torch.tensor(S_true_np, dtype=torch.float32, device=device),
        "area_src": torch.tensor(area_src_np, dtype=torch.float32, device=device),
        "area_tgt": torch.tensor(area_tgt_np, dtype=torch.float32, device=device),
        "n_src": n_src,
        "n_tgt": n_tgt,
        "n_edges": len(df),
        "n_pos": float(edge_exists_np.sum()),
    }

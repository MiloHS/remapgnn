from __future__ import annotations

from pathlib import Path
import argparse
import re
import subprocess
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from remapgnn.config import load_config


def safe_pair_name(pair: str) -> str:
    return pair.replace("-", "").replace("_to_", "_to_")


def ensure_decoder_chunk_size(s: str, chunk_size: int) -> str:
    if re.search(r"^DECODER_CHUNK_SIZE\s*=", s, flags=re.MULTILINE):
        return re.sub(
            r"^DECODER_CHUNK_SIZE\s*=.*",
            f"DECODER_CHUNK_SIZE = {chunk_size}",
            s,
            flags=re.MULTILINE,
        )

    if re.search(r"^BALANCE_ITERS\s*=", s, flags=re.MULTILINE):
        return re.sub(
            r"^(BALANCE_ITERS\s*=.*\n)",
            rf"\1DECODER_CHUNK_SIZE = {chunk_size}\n",
            s,
            count=1,
            flags=re.MULTILINE,
        )

    return re.sub(
        r"^(K\s*=\s*['\"][^'\"]+['\"]\n)",
        rf"\1DECODER_CHUNK_SIZE = {chunk_size}\n",
        s,
        count=1,
        flags=re.MULTILINE,
    )


def make_eval_script(cfg, pair: str) -> Path:
    template = Path(cfg.raw["paths"]["eval_template"])
    if not template.exists():
        raise FileNotFoundError(f"eval template not found: {template}")

    s = template.read_text()

    # Core config substitutions.
    s = re.sub(r'^PAIR\s*=.*', f'PAIR = "{pair}"', s, count=1, flags=re.MULTILINE)
    s = re.sub(r'^K\s*=.*', f'K = "{cfg.K}"', s, count=1, flags=re.MULTILINE)

    bal_iters = int(cfg.raw.get("training", {}).get("balance_iters", 2000))
    s = re.sub(r'^BALANCE_ITERS\s*=.*', f'BALANCE_ITERS = {bal_iters}', s, count=1, flags=re.MULTILINE)

    chunk_size = int(cfg.raw.get("training", {}).get("decoder_chunk_size", 10000))
    s = ensure_decoder_chunk_size(s, chunk_size)

    s = re.sub(
        r'^GNN_PATH\s*=.*',
        f'GNN_PATH = Path("{cfg.model_path}")',
        s,
        count=1,
        flags=re.MULTILINE,
    )

    s = re.sub(
        r'^EDGE_PATH\s*=.*',
        'EDGE_PATH = Path(f"analysis_medium_improv/edge_dataset_{PAIR}_k{K}.parquet")',
        s,
        count=1,
        flags=re.MULTILINE,
    )

    s = re.sub(
        r'^MAP_PATH\s*=.*',
        'MAP_PATH = Path(f"maps_medium_improv/map_{PAIR}_conserve.nc")',
        s,
        count=1,
        flags=re.MULTILINE,
    )

    # Ensure field files follow pair.
    if "SRC, TGT = PAIR.split" not in s:
        s = re.sub(
            r'^(MAP_PATH\s*=.*\n)',
            r'\1SRC, TGT = PAIR.split("_to_")\n',
            s,
            count=1,
            flags=re.MULTILINE,
        )

    s = re.sub(
        r'^SRC_FILE\s*=.*',
        'SRC_FILE = MIRA_DIR / f"Meshes/UniformlyRefined/{SRC.split(\'-\')[0]}/sample_NM16_O10_{SRC}_TPW_CFR_TPO_A1_A2.nc"',
        s,
        count=1,
        flags=re.MULTILINE,
    )

    s = re.sub(
        r'^TGT_FILE\s*=.*',
        'TGT_FILE = MIRA_DIR / f"Meshes/UniformlyRefined/{TGT.split(\'-\')[0]}/sample_NM16_O10_{TGT}_TPW_CFR_TPO_A1_A2.nc"',
        s,
        count=1,
        flags=re.MULTILINE,
    )

    # Clean stale printed label if present.
    s = s.replace(
        "Bipartite GNN v4 GPU chunked + Sinkhorn vs Tempest:",
        f"{cfg.run_name} + sparse Sinkhorn vs Tempest:",
    )

    out_dir = ROOT / ".generated"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"evaluate_{cfg.run_name}_{safe_pair_name(pair)}.py"
    out.write_text(s)
    return out





def patch_generated_for_framework_model(script, cfg):
    """
    Patch legacy generated evaluator scripts so config/framework models work:
      - import build_model()
      - add synthetic mesh-family conditioning columns when needed
      - use checkpoint feature lists instead of legacy hardcoded feature lists
      - replace legacy model construction with build_model()
      - clean v10 hardcoded method labels
    """
    text = script.read_text()

    if "from remapgnn.models import build_model" not in text:
        import_block = "\n".join([
            "import sys",
            "from pathlib import Path",
            "ROOT = Path(__file__).resolve().parents[1]",
            "if str(ROOT) not in sys.path:",
            "    sys.path.insert(0, str(ROOT))",
            "from remapgnn.models import build_model",
            "",
            "def _mesh_family_name_for_eval(x):",
            "    x = str(x).upper()",
            "    if 'RLL' in x:",
            "        return 'RLL'",
            "    if 'ICOD' in x:",
            "        return 'ICOD'",
            "    if 'CS' in x:",
            "        return 'CS'",
            "    return 'OTHER'",
            "",
            "def _add_mesh_condition_columns_for_eval(df):",
            "    df = df.copy()",
            "    if 'src_mesh' in df.columns:",
            "        src_family = df['src_mesh'].map(_mesh_family_name_for_eval)",
            "    elif 'pair' in df.columns:",
            "        src_family = df['pair'].astype(str).str.split('_to_').str[0].map(_mesh_family_name_for_eval)",
            "    else:",
            "        src_family = None",
            "    if 'tgt_mesh' in df.columns:",
            "        tgt_family = df['tgt_mesh'].map(_mesh_family_name_for_eval)",
            "    elif 'pair' in df.columns:",
            "        tgt_family = df['pair'].astype(str).str.split('_to_').str[1].map(_mesh_family_name_for_eval)",
            "    else:",
            "        tgt_family = None",
            "    for fam in ['RLL', 'CS', 'ICOD']:",
            "        if src_family is not None:",
            "            df[f'src_mesh_is_{fam}'] = (src_family == fam).astype('float32')",
            "        if tgt_family is not None:",
            "            df[f'tgt_mesh_is_{fam}'] = (tgt_family == fam).astype('float32')",
            "    return df",
            "",
        ])
        text = import_block + "\n" + text

    lines = text.splitlines(True)

    # Add mesh-conditioning columns immediately after reading the edge dataframe.
    new_lines = []
    added_mesh_columns = False
    for line in lines:
        new_lines.append(line)
        stripped = line.strip()
        if (
            not added_mesh_columns
            and "pd.read_parquet" in stripped
            and "EDGE_PATH" in stripped
            and "=" in stripped
        ):
            lhs = stripped.split("=", 1)[0].strip()
            indent = line[: len(line) - len(line.lstrip())]
            if lhs:
                new_lines.append(f"{indent}{lhs} = _add_mesh_condition_columns_for_eval({lhs})\n")
                added_mesh_columns = True
    lines = new_lines

    # Replace legacy feature extraction with checkpoint/config feature lists.
    new_lines = []
    inserted_edge_features = False
    inserted_src_features = False
    inserted_tgt_features = False

    for line in lines:
        indent = line[: len(line) - len(line.lstrip())]

        if "[EDGE_FEATURES]" in line and ".to_numpy" in line:
            if not inserted_edge_features:
                new_lines.append(f'{indent}edge_features = pack.get("edge_features", EDGE_FEATURES)\n')
                inserted_edge_features = True
            line = line.replace("[EDGE_FEATURES]", "[edge_features]")

        if "[SRC_NODE_FEATURES]" in line and ".to_numpy" in line:
            if not inserted_src_features:
                new_lines.append(f'{indent}src_node_features = pack.get("src_node_features", SRC_NODE_FEATURES)\n')
                inserted_src_features = True
            line = line.replace("[SRC_NODE_FEATURES]", "[src_node_features]")

        if "[TGT_NODE_FEATURES]" in line and ".to_numpy" in line:
            if not inserted_tgt_features:
                new_lines.append(f'{indent}tgt_node_features = pack.get("tgt_node_features", TGT_NODE_FEATURES)\n')
                inserted_tgt_features = True
            line = line.replace("[TGT_NODE_FEATURES]", "[tgt_node_features]")

        new_lines.append(line)

    lines = new_lines

    # Replace legacy model construction.
    model_start = None
    for i, line in enumerate(lines):
        if "model" in line and "BipartiteGNNSinkhorn(" in line:
            model_start = i
            break

    if model_start is None:
        print("WARNING: did not find legacy BipartiteGNNSinkhorn model construction to patch")
        patched = "".join(lines)
    else:
        indent = lines[model_start][: len(lines[model_start]) - len(lines[model_start].lstrip())]

        depth = 0
        model_end = model_start
        for j in range(model_start, len(lines)):
            depth += lines[j].count("(") - lines[j].count(")")
            model_end = j
            if j > model_start and depth <= 0:
                if ".to(device)" in lines[j] or ".to(eval_device)" in lines[j]:
                    pass
                elif j + 1 < len(lines) and (".to(device)" in lines[j + 1] or ".to(eval_device)" in lines[j + 1]):
                    model_end = j + 1
                break

        replacement = [
            f'{indent}eval_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n',
            f'{indent}model = build_model(\n',
            f'{indent}    architecture=pack.get("architecture", "{cfg.architecture}"),\n',
            f'{indent}    src_dim=len(pack["src_node_features"]),\n',
            f'{indent}    tgt_dim=len(pack["tgt_node_features"]),\n',
            f'{indent}    edge_dim=len(pack["edge_features"]),\n',
            f'{indent}    hidden=int(pack.get("hidden", 128)),\n',
            f'{indent}    decoder_chunk_size=int(pack.get("decoder_chunk_size", DECODER_CHUNK_SIZE)),\n',
            f'{indent}).to(eval_device)\n',
        ]

        lines = lines[:model_start] + replacement + lines[model_end + 1 :]
        patched = "".join(lines)

    # Cosmetic cleanup: legacy evaluator templates hard-code v10 method labels.
    legacy_labels = [
        "bipartite_gnn_v10_hybridattn_kdist_a2p0_mink8",
        "bipartite_gnn_sinkhorn_v10_hybridattn_kdist_a2p0_mink8",
    ]
    for legacy in legacy_labels:
        patched = patched.replace(legacy, cfg.model_tag)

    script.write_text(patched)
    print(f"patched generated evaluator architecture: {cfg.architecture}")
    print(f"patched generated evaluator label: {cfg.model_tag}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pair")
    group.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pairs = cfg.pairs if args.all_pairs else [args.pair]

    for pair in pairs:
        script = make_eval_script(cfg, pair)
        print(f"\nRUNNING {pair}")
        print(f"generated: {script}")
        patch_generated_for_framework_model(script, cfg)

        if not args.dry_run:
            run_start_time = time.time()
            subprocess.run([sys.executable, "-u", str(script)], check=True)

            # Some legacy evaluator templates still write results using their
            # original hard-coded model tag. Move only files created by this run.
            expected_eval = cfg.eval_csv(pair)
            expected_diag = cfg.diagnostics_csv(pair)

            # If the legacy template wrote the correct expected names, do nothing.
            # Otherwise find fresh files for this pair and move them into config paths.
            fresh_eval_candidates = sorted(
                [
                    x for x in cfg.analysis_dir.glob(f"*eval_{pair}_{cfg.graph_suffix}.csv")
                    if x != expected_eval and x.stat().st_mtime >= run_start_time
                ],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )
            fresh_diag_candidates = sorted(
                [
                    x for x in cfg.analysis_dir.glob(f"*operator_diagnostics_{pair}_{cfg.graph_suffix}.csv")
                    if x != expected_diag and x.stat().st_mtime >= run_start_time
                ],
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

            if not expected_eval.exists() and fresh_eval_candidates:
                candidate = fresh_eval_candidates[0]
                expected_eval.parent.mkdir(parents=True, exist_ok=True)
                candidate.replace(expected_eval)
                print(f"moved eval csv: {candidate} -> {expected_eval}")

            if not expected_diag.exists() and fresh_diag_candidates:
                candidate = fresh_diag_candidates[0]
                expected_diag.parent.mkdir(parents=True, exist_ok=True)
                candidate.replace(expected_diag)
                print(f"moved diagnostics: {candidate} -> {expected_diag}")

if __name__ == "__main__":
    main()

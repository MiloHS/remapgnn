from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class RunConfig:
    raw: dict
    path: Path

    @property
    def run_name(self) -> str:
        return self.raw["run_name"]

    @property
    def model_tag(self) -> str:
        return self.raw["model_tag"]

    @property
    def architecture(self) -> str:
        return self.raw["architecture"]

    @property
    def graph_suffix(self) -> str:
        return self.raw["graph"]["suffix"]

    @property
    def K(self) -> str:
        return self.raw["graph"]["K"]

    @property
    def pairs(self) -> list[str]:
        return list(self.raw["pairs"])

    @property
    def analysis_dir(self) -> Path:
        return Path(self.raw["paths"]["analysis_dir"])

    @property
    def maps_dir(self) -> Path:
        return Path(self.raw["paths"]["maps_dir"])

    @property
    def models_dir(self) -> Path:
        return Path(self.raw["paths"]["models_dir"])

    @property
    def mira_dir(self) -> Path:
        return Path(self.raw["paths"]["mira_dir"])

    @property
    def model_path(self) -> Path:
        return self.models_dir / f"{self.model_tag}.pt"

    @property
    def history_path(self) -> Path:
        return self.models_dir / f"{self.model_tag}_history.csv"

    def edge_path(self, pair: str) -> Path:
        return self.analysis_dir / f"edge_dataset_{pair}_{self.graph_suffix}.parquet"

    def map_path(self, pair: str) -> Path:
        return self.maps_dir / f"map_{pair}_conserve.nc"

    def eval_csv(self, pair: str) -> Path:
        return self.analysis_dir / f"{self.model_tag}_eval_{pair}_{self.graph_suffix}.csv"

    def diagnostics_csv(self, pair: str) -> Path:
        return self.analysis_dir / f"{self.model_tag}_operator_diagnostics_{pair}_{self.graph_suffix}.csv"

    def source_target_files(self, pair: str) -> tuple[Path, Path]:
        src, tgt = pair.split("_to_")
        src_mesh = src.split("-")[0]
        tgt_mesh = tgt.split("-")[0]
        src_file = self.mira_dir / f"Meshes/UniformlyRefined/{src_mesh}/sample_NM16_O10_{src}_TPW_CFR_TPO_A1_A2.nc"
        tgt_file = self.mira_dir / f"Meshes/UniformlyRefined/{tgt_mesh}/sample_NM16_O10_{tgt}_TPW_CFR_TPO_A1_A2.nc"
        return src_file, tgt_file


def load_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with path.open("r") as f:
        raw = json.load(f)
    return RunConfig(raw=raw, path=path)

"""Analysis tables for the notebook, built from saved run artifacts.

Every function here reads what training already wrote -- ``metrics.json``,
``curve.csv``, ``predictions.npz``, ``diagnostics.npz``, ``controls.json`` --
and returns a ``pandas.DataFrame`` the notebook can display as-is. Nothing here
trains, evaluates or touches a GPU.

Why a module rather than notebook cells. The Kaggle notebook has to tell the
whole story of the experiment, which means a dozen analysis sections. Written
inline, each one becomes twenty lines of untested pandas that silently produces
the wrong table when a column is renamed. Written here, each is one function
with one test, and the notebook cell is a single call.

Every function tolerates a missing artifact: an absent run, a baseline that was
skipped, diagnostics that were not dumped. It returns an empty frame (or ``None``)
and says why, rather than raising -- a notebook that dies in section 11 because
one optional file is missing has thrown away sections 1-10.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src import bootstrap
from src.metrics import Predictions, masked_bias, masked_mae, masked_rmse, top_decile_mask

HORIZONS = (1, 6, 12, 24)

GRID = [
    "t0_temporal_only",
    "s0_static_concat",
    "d0_wind_concat",
    "s1_static_crossview",
    "d1_wind_crossview",
]
BASELINES = [
    "persistence",
    "climatology",
    "seasonal_naive_24h",
    "seasonal_naive_168h",
    "ridge",
    "dlinear",
    "lightgbm",
]

# What each grid cell is, in the terms the thesis uses. The comparison column is
# the question that config exists to answer.
GRID_DESIGN = [
    ("t0_temporal_only", "none", "none", "none", "is a graph needed at all?"),
    ("s0_static_concat", "static distance", "GCN", "concat", "conventional reference"),
    ("d0_wind_concat", "lag-aware wind", "wind GAT", "concat", "s0 -> d0: the graph, alone"),
    ("s1_static_crossview", "static distance", "GCN", "cross-view", "s0 -> s1: the fusion, alone"),
    ("d1_wind_crossview", "lag-aware wind", "wind GAT", "cross-view", "full model"),
]


@dataclass(frozen=True)
class RunInfo:
    name: str
    seed: int
    path: Path


# ------------------------------------------------------------------ discovery
def list_runs(runs_root: Path | str) -> list[RunInfo]:
    """Every ``<name>_seed<N>`` directory with a metrics file, sorted."""
    root = Path(runs_root)
    found: list[RunInfo] = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        stem = metrics_path.parent.name
        if "_seed" not in stem:
            continue
        name, _, seed = stem.rpartition("_seed")
        try:
            found.append(RunInfo(name=name, seed=int(seed), path=metrics_path.parent))
        except ValueError:
            continue
    return found


def _metrics(run: RunInfo) -> dict[str, Any]:
    with open(run.path / "metrics.json", encoding="utf-8") as handle:
        return json.load(handle)


def _rows(runs_root: Path | str, names: Sequence[str] | None = None) -> pd.DataFrame:
    """Tidy (model, seed, horizon, mae, rmse, wape, bias) rows across every run."""
    records = []
    for run in list_runs(runs_root):
        if names is not None and run.name not in names:
            continue
        for row in _metrics(run).get("rows", []):
            records.append(row)
    if not records:
        return pd.DataFrame(columns=["model", "seed", "horizon", "mae", "rmse", "wape", "bias"])
    return pd.DataFrame(records)


# ------------------------------------------------------------- 1. configuration
def grid_design() -> pd.DataFrame:
    """The five configurations and the single question each one isolates."""
    return pd.DataFrame(
        GRID_DESIGN, columns=["config", "graph", "spatial", "fusion", "isolates"]
    ).set_index("config")


def resolved_config(config_path: Path | str = "configs/model/d1.yaml") -> dict[str, Any]:
    """The full config a run actually used, base merged with the model file."""
    from src.train import load_config

    return load_config(Path(config_path))


def flatten_config(config: dict[str, Any], prefix: str = "") -> pd.DataFrame:
    """A nested config as a two-column (setting, value) table."""
    items: list[tuple[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        else:
            items.append((path, node))

    walk(config, prefix)
    return pd.DataFrame(items, columns=["setting", "value"]).set_index("setting")


def environment_table(runs_root: Path | str) -> pd.DataFrame:
    """What the runs executed on, from the first env.json found."""
    for run in list_runs(runs_root):
        env_path = run.path / "env.json"
        if not env_path.exists():
            continue
        with open(env_path, encoding="utf-8") as handle:
            env = json.load(handle)
        rows = {
            "git revision": env.get("git_revision"),
            "python": env.get("python"),
            "platform": env.get("platform"),
            "timestamp (UTC)": env.get("timestamp_utc"),
        }
        torch_info = env.get("torch") or {}
        rows["torch"] = torch_info.get("version")
        rows["cuda"] = torch_info.get("cuda_version")
        rows["device"] = torch_info.get("device_name") or "cpu"
        for name, version in (env.get("packages") or {}).items():
            rows[name] = version
        return pd.DataFrame.from_dict(rows, orient="index", columns=["value"])
    return pd.DataFrame(columns=["value"])


# ------------------------------------------------------------------ 2. dataset
def dataset_summary(dataset, lookback: int = 48, horizon: int = 24) -> pd.DataFrame:
    """Splits with dates, hours, forecast windows, and target statistics.

    The window count is the real one -- origins whose full target lies inside
    the split and has at least one observed value -- not ``hours - horizon``.
    It is the number every model was actually scored on.
    """
    from src.data.windows import valid_origins

    rows = []
    for split in dataset.splits.values():
        mask = dataset.target_mask[split.start : split.end]
        target = dataset.target_raw[split.start : split.end]
        observed = target[mask]
        origins = valid_origins(
            split=split, n_hours=dataset.n_hours, lookback=lookback,
            horizon=horizon, target_mask=dataset.target_mask,
        )
        rows.append(
            {
                "split": split.name,
                "start": split.start_date,
                "end": split.end_date,
                "hours": split.end - split.start,
                f"windows ({lookback}h -> {horizon}h)": int(len(origins)),
                "PM2.5 observed": f"{mask.mean() * 100:.2f} %",
                "mean PM2.5": round(float(observed.mean()), 1) if observed.size else np.nan,
                "median": round(float(np.median(observed)), 1) if observed.size else np.nan,
                "p95": round(float(np.percentile(observed, 95)), 0) if observed.size else np.nan,
                "> 150 ug/m3": f"{(observed > 150).mean() * 100:.1f} %" if observed.size else "",
            }
        )
    return pd.DataFrame(rows).set_index("split")


def feature_table(dataset) -> pd.DataFrame:
    """Every model input, its transform, and the training-split statistics."""
    rows = []
    for spec in dataset.scaler.specs:
        rows.append(
            {
                "feature": spec.name,
                "transform": spec.kind,
                "mean (fit)": round(spec.mean, 3) if spec.kind != "none" else "",
                "std (fit)": round(spec.std, 3) if spec.kind != "none" else "",
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def station_missingness(dataset) -> pd.DataFrame:
    """Observed PM2.5 fraction and mean level per station, whole record."""
    rows = []
    for index, name in enumerate(dataset.stations):
        mask = dataset.target_mask[:, index]
        values = dataset.target_raw[:, index][mask]
        rows.append(
            {
                "station": name,
                "lat": round(float(dataset.coords[index, 0]), 4),
                "lon": round(float(dataset.coords[index, 1]), 4),
                "PM2.5 observed": f"{mask.mean() * 100:.2f} %",
                "mean PM2.5": round(float(values.mean()), 1) if values.size else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("station")


# ------------------------------------------------------------ 3. architecture
def architecture_table(dataset, model_dir: Path | str = "configs/model") -> pd.DataFrame:
    """Parameter count per component for each configuration in the grid.

    Built on the fly from the YAML, so it reflects the configs as committed
    rather than a number typed into a table by hand.
    """
    from src.models.sdgt import build_model
    from src.train import load_config, model_config_from

    rows = []
    for stem in ("t0", "s0", "d0", "s1", "d1"):
        path = Path(model_dir) / f"{stem}.yaml"
        if not path.exists():
            continue
        config = load_config(path)
        model = build_model(model_config_from(config, dataset))
        counts = model.parameter_counts()
        rows.append({"config": config["name"], **counts})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("config")
    return frame[["embedding", "temporal", "graph", "spatial", "fusion", "head", "total"]]


# ------------------------------------------------------ 4/5. training curves
def training_table(runs_root: Path | str, names: Sequence[str] = GRID) -> pd.DataFrame:
    """Convergence facts per run: epochs, best epoch, final train/val ratio, time.

    The train/val ratio is the single number that diagnosed Run 1: at 3-4x every
    model was memorising. Near 1.0 is the target; well below 1.0 means dropout
    and weight decay are doing more than the data warrants.
    """
    rows = []
    for run in list_runs(runs_root):
        if run.name not in names:
            continue
        metrics = _metrics(run)
        curve = load_curve(run.path)
        final = curve.iloc[-1] if len(curve) else None
        ratio = (
            float(final["val_loss"]) / float(final["train_loss"])
            if final is not None and float(final["train_loss"]) > 0
            else np.nan
        )
        rows.append(
            {
                "model": run.name,
                "seed": run.seed,
                "epochs run": metrics.get("epochs_run"),
                "best epoch": metrics.get("best_epoch"),
                "best val MAE": round(float(metrics.get("best_val_mae", np.nan)), 2),
                "final train loss": round(float(final["train_loss"]), 4) if final is not None else np.nan,
                "final val loss": round(float(final["val_loss"]), 4) if final is not None else np.nan,
                "val / train": round(ratio, 2),
                "minutes": round(float(metrics.get("train_seconds", 0.0)) / 60.0, 1),
                "params": (metrics.get("parameters") or {}).get("total"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["model", "seed"]).sort_index()


def load_curve(run_dir: Path | str) -> pd.DataFrame:
    path = Path(run_dir) / "curve.csv"
    if not path.exists():
        return pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "val_mae", "lr", "seconds"])
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    frame = pd.DataFrame(rows)
    return frame.apply(pd.to_numeric, errors="coerce")


# --------------------------------------------------------- 6. quantitative
def metrics_table(
    runs_root: Path | str,
    metric: str = "mae",
    names: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """One row per model, one column per horizon, ``mean ± sd`` across seeds.

    Single-seed models (the baselines) show the bare number. Bold nothing: the
    seed spread is the first thing to compare against before any difference is
    called real.
    """
    rows = _rows(runs_root, names)
    if seeds is not None:
        rows = rows[rows["seed"].isin(list(seeds))]
    if rows.empty:
        return pd.DataFrame()

    def cell(values: pd.Series) -> str:
        if len(values) > 1:
            return f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"
        return f"{values.iloc[0]:.2f}"

    table = (
        rows.groupby(["model", "horizon"])[metric]
        .apply(cell)
        .unstack("horizon")
    )
    table.columns = [f"h={h}" for h in table.columns]
    order = [n for n in BASELINES + GRID if n in table.index]
    return table.loc[order + [n for n in table.index if n not in order]]


def metrics_numeric(
    runs_root: Path | str, metric: str = "mae", names: Sequence[str] | None = None
) -> pd.DataFrame:
    """Same as :func:`metrics_table` but numeric means, for arithmetic."""
    rows = _rows(runs_root, names)
    if rows.empty:
        return pd.DataFrame()
    table = rows.groupby(["model", "horizon"])[metric].mean().unstack("horizon")
    table.columns = [f"h={h}" for h in table.columns]
    return table


# ------------------------------------------------- 7. baseline comparison
def skill_vs_persistence(runs_root: Path | str) -> pd.DataFrame:
    """Percentage MAE improvement over persistence, per model and horizon.

    Skill = 1 - MAE_model / MAE_persistence. Zero is "no better than repeating
    the last observation"; negative is worse than that. This is the standard
    forecasting yardstick and it is what makes h=1 honest: everything sits near
    zero there because persistence is already very hard to beat.
    """
    numeric = metrics_numeric(runs_root, "mae")
    if numeric.empty or "persistence" not in numeric.index:
        return pd.DataFrame()
    reference = numeric.loc["persistence"]
    skill = (1.0 - numeric.div(reference, axis=1)) * 100.0
    skill = skill.drop(index="persistence")
    order = [n for n in BASELINES + GRID if n in skill.index]
    return skill.loc[order + [n for n in skill.index if n not in order]].round(1)


def _load_predictions(run: RunInfo) -> Predictions | None:
    path = run.path / "predictions.npz"
    return Predictions.load(path) if path.exists() else None


def paired_vs_reference(
    runs_root: Path | str,
    reference: str = "lightgbm",
    candidates: Sequence[str] = GRID,
    seeds: Sequence[int] | None = None,
    n_boot: int = 400,
) -> pd.DataFrame:
    """Paired weekly block bootstrap of every grid config against one baseline.

    Positive means the candidate beats the reference. A verdict is ``robust``
    only when every seed agrees in sign and every seed's interval excludes
    zero -- the same standard the grid's internal comparisons are held to.
    """
    runs = list_runs(runs_root)
    reference_runs = [r for r in runs if r.name == reference]
    if not reference_runs:
        return pd.DataFrame()
    ref_pred = _load_predictions(reference_runs[0])
    if ref_pred is None:
        return pd.DataFrame()

    rows = []
    for name in candidates:
        cand_runs = [r for r in runs if r.name == name and (seeds is None or r.seed in seeds)]
        preds = [p for p in (_load_predictions(r) for r in cand_runs) if p is not None]
        if not preds:
            continue
        for h in HORIZONS:
            intervals = [
                bootstrap.paired_interval(ref_pred, p, horizon=h, n_boot=n_boot, seed=p.seed)
                for p in preds
            ]
            points = np.array([i.point for i in intervals])
            n_sig = sum(i.significant for i in intervals)
            same_sign = bool((points > 0).all() or (points < 0).all())
            if n_sig == len(intervals) and same_sign:
                verdict = "robust"
            elif n_sig:
                verdict = "seed-dependent"
            else:
                verdict = "comparable"
            rows.append(
                {
                    "candidate": name,
                    "horizon": h,
                    "diff (ug/m3)": round(float(points.mean()), 3),
                    "seed sd": round(float(points.std(ddof=1)), 3) if len(points) > 1 else 0.0,
                    "CIs excl. 0": f"{n_sig}/{len(intervals)}",
                    "verdict": verdict,
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index(["candidate", "horizon"])
    frame.attrs["reference"] = reference
    return frame


# -------------------------------------------------------- 9. station-wise
def station_table(
    dataset, runs_root: Path | str, model: str, horizon: int = 24, seed: int | None = None
) -> pd.DataFrame:
    """Per-station MAE for one model, next to the station's own difficulty.

    A station is hard because its level is high or its record is gappy, not
    because the model singled it out. Showing missingness and mean level beside
    the error lets the reader tell those apart.
    """
    run = _pick_run(runs_root, model, seed)
    if run is None:
        return pd.DataFrame()
    pred = _load_predictions(run)
    if pred is None:
        return pd.DataFrame()

    persistence = _pick_run(runs_root, "persistence", None)
    pers_pred = _load_predictions(persistence) if persistence else None

    index = horizon - 1
    rows = []
    for s, name in enumerate(dataset.stations):
        mask = pred.mask[:, s, index]
        mae = masked_mae(pred.pred[:, s, index], pred.truth[:, s, index], mask)
        bias = masked_bias(pred.pred[:, s, index], pred.truth[:, s, index], mask)
        row = {
            "station": name,
            f"MAE h={horizon}": round(mae, 2),
            "bias": round(bias, 2),
            "mean observed": round(float(pred.truth[:, s, index][mask].mean()), 1) if mask.any() else np.nan,
            "observed": f"{mask.mean() * 100:.1f} %",
        }
        if pers_pred is not None:
            pmae = masked_mae(pers_pred.pred[:, s, index], pers_pred.truth[:, s, index], mask)
            row["persistence MAE"] = round(pmae, 2)
            row["skill %"] = round((1 - mae / pmae) * 100, 1) if pmae > 0 else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("station")
    frame.attrs["model"] = run.name
    frame.attrs["seed"] = run.seed
    return frame.sort_values(f"MAE h={horizon}", ascending=False)


def _pick_run(runs_root: Path | str, model: str, seed: int | None) -> RunInfo | None:
    matches = [r for r in list_runs(runs_root) if r.name == model]
    if seed is not None:
        matches = [r for r in matches if r.seed == seed]
    if not matches:
        return None
    return min(matches, key=lambda r: r.seed)


# --------------------------------------------------- 10. dynamic graph
def graph_summary(runs_root: Path | str) -> pd.DataFrame:
    """Learned graph physics from every run that dumped ``graph_diagnostics.json``.

    ``prior_strength`` and ``decay_rate`` both start at 1.0. How far they moved
    is the difference between "the graph trained" and "the graph sat at its
    initialisation" -- which is exactly what separated Run 2 from Run 1.
    """
    rows = []
    for run in list_runs(runs_root):
        path = run.path / "graph_diagnostics.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "model": run.name,
                "seed": run.seed,
                "prior strength (init 1.0)": round(payload.get("prior_strength", np.nan), 3),
                "decay rate (init 1.0)": round(payload.get("graph_decay_rate", np.nan), 3),
                "mean transport lag (h)": round(payload.get("graph_mean_lag", np.nan), 2),
                "mean degree": round(payload.get("graph_mean_degree", np.nan), 2),
                "calm fallback rate": f"{payload.get('graph_fallback_rate', np.nan) * 100:.1f} %",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["model", "seed"])


def load_diagnostics(runs_root: Path | str, model: str = "d1_wind_crossview") -> dict | None:
    """The sampled diagnostics arrays for one model, lowest seed that has them."""
    for run in sorted((r for r in list_runs(runs_root) if r.name == model), key=lambda r: r.seed):
        path = run.path / "diagnostics.npz"
        if path.exists():
            with np.load(path) as payload:
                arrays = {key: payload[key] for key in payload.files}
            arrays["_run"] = run.path.name
            return arrays
    return None


def adjacency_by_regime(diagnostics: dict, calm_threshold: float = 1.5) -> pd.DataFrame:
    """How the graph differs between calm and windy windows.

    The physical claim is that the wind graph *changes* with the wind. If the
    adjacency in a 6 m/s hour looks like the adjacency in a 0.5 m/s hour, the
    graph is static in practice whatever it is on paper.
    """
    if "adjacency" not in diagnostics or "wind_speed" not in diagnostics:
        return pd.DataFrame()
    adjacency = diagnostics["adjacency"]              # [n, L, N, N]
    speed = diagnostics["wind_speed"]                 # [n, L]
    calm_mask = diagnostics.get("graph_calm")         # [n, L, N, 1] or None

    n_stations = adjacency.shape[-1]
    eye = np.eye(n_stations, dtype=bool)

    rows = []
    for label, selector in (
        (f"calm (< {calm_threshold} m/s)", speed < calm_threshold),
        (f"windy (>= {calm_threshold} m/s)", speed >= calm_threshold),
    ):
        if not selector.any():
            continue
        adj = adjacency[selector]                     # [k, N, N]
        off = adj[:, ~eye]
        self_weight = adj[:, eye]
        active = (adj > 1e-4)
        rows.append(
            {
                "regime": label,
                "hours": int(selector.sum()),
                "mean self-weight": round(float(self_weight.mean()), 3),
                "mean off-diagonal weight": round(float(off.mean()), 4),
                "mean degree": round(float(active.sum(-1).mean()), 2),
                "fallback to static": (
                    f"{float(calm_mask[selector].mean()) * 100:.1f} %" if calm_mask is not None else ""
                ),
            }
        )
    return pd.DataFrame(rows).set_index("regime")


# ----------------------------------------------- 11. attention and fusion
def attention_by_lag(diagnostics: dict) -> pd.DataFrame:
    """Share of cross-station attention mass at each transport lag.

    The whole point of the lag-aware design is that a source station's state
    *k* hours ago is what should matter, with *k* set by distance and wind. The
    evidence review measured the real advection signal peaking at a 2 h lag. If
    the learned attention piles up at lag 0 instead, the model is using the
    graph as a same-hour association, not as transport.
    """
    if "attention_by_lag" not in diagnostics:
        return pd.DataFrame()
    weights = diagnostics["attention_by_lag"]          # [n, L, N(tgt), N(src), K+1]
    n_stations = weights.shape[2]
    eye = np.eye(n_stations, dtype=bool)
    cross = weights[:, :, ~eye, :]                      # drop self-attention
    by_lag = cross.sum(axis=(0, 1, 2))
    share = by_lag / max(by_lag.sum(), 1e-12)
    frame = pd.DataFrame(
        {"lag (h)": np.arange(len(share)), "attention share": np.round(share * 100, 1)}
    ).set_index("lag (h)")
    frame.attrs["peak_lag"] = int(np.argmax(share))
    return frame


def gate_summary(diagnostics: dict) -> pd.DataFrame:
    """Where the cross-view gate sits, overall and per patch position.

    0 = pure temporal, 1 = pure spatial, 0.5 = the initialisation. A gate that
    never leaves 0.5 has not learned to choose; a gate with real spread has.
    """
    if "gate" not in diagnostics:
        return pd.DataFrame()
    gate = diagnostics["gate"]                          # [n, N, Np, d]
    rows = [
        {
            "scope": "overall",
            "mean": round(float(gate.mean()), 3),
            "std": round(float(gate.std()), 3),
            "min": round(float(gate.min()), 3),
            "max": round(float(gate.max()), 3),
            "share > 0.5 (leans spatial)": f"{float((gate > 0.5).mean()) * 100:.1f} %",
        }
    ]
    n_patches = gate.shape[2]
    for p in range(n_patches):
        slab = gate[:, :, p, :]
        if p == 0:
            label = f"patch {p} (oldest)"
        elif p == n_patches - 1:
            label = f"patch {p} (newest)"
        else:
            label = f"patch {p}"
        rows.append(
            {
                "scope": label,
                "mean": round(float(slab.mean()), 3),
                "std": round(float(slab.std()), 3),
                "min": round(float(slab.min()), 3),
                "max": round(float(slab.max()), 3),
                "share > 0.5 (leans spatial)": f"{float((slab > 0.5).mean()) * 100:.1f} %",
            }
        )
    return pd.DataFrame(rows).set_index("scope")


# ------------------------------------------------ 12. error and failure
def bias_by_horizon(runs_root: Path | str, names: Sequence[str] | None = None) -> pd.DataFrame:
    """Mean signed error per horizon: negative means the model under-forecasts.

    MAE hides direction. A model that is systematically low on episodes will
    show a negative bias that grows with horizon, which is the signature of
    regression to the mean.
    """
    rows = _rows(runs_root, names)
    if rows.empty:
        return pd.DataFrame()
    table = rows.groupby(["model", "horizon"])["bias"].mean().unstack("horizon").round(2)
    table.columns = [f"h={h}" for h in table.columns]
    order = [n for n in BASELINES + GRID if n in table.index]
    return table.loc[order + [n for n in table.index if n not in order]]


def error_by_level(
    runs_root: Path | str, model: str, horizon: int = 24, n_bins: int = 5, seed: int | None = None
) -> pd.DataFrame:
    """MAE and bias inside quantile bins of the *observed* concentration.

    This is where an early-warning system is actually judged. A model can have
    a fine overall MAE and still be useless on the top decile, because the top
    decile is where it regresses toward the mean.
    """
    run = _pick_run(runs_root, model, seed)
    if run is None:
        return pd.DataFrame()
    pred = _load_predictions(run)
    if pred is None:
        return pd.DataFrame()

    index = horizon - 1
    p, t, m = pred.pred[:, :, index], pred.truth[:, :, index], pred.mask[:, :, index]
    observed = t[m]
    if observed.size == 0:
        return pd.DataFrame()
    edges = np.quantile(observed, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.inf

    persistence = _pick_run(runs_root, "persistence", None)
    pers = _load_predictions(persistence) if persistence else None

    rows = []
    for i in range(n_bins):
        inside = m & (t >= edges[i]) & (t < edges[i + 1])
        if not inside.any():
            continue
        row = {
            "observed PM2.5 bin": f"{edges[i]:.0f} - {edges[i + 1]:.0f}" if np.isfinite(edges[i + 1]) else f">= {edges[i]:.0f}",
            "n": int(inside.sum()),
            "MAE": round(masked_mae(p, t, inside), 2),
            "bias": round(masked_bias(p, t, inside), 2),
            "RMSE": round(masked_rmse(p, t, inside), 2),
        }
        if pers is not None:
            row["persistence MAE"] = round(
                masked_mae(pers.pred[:, :, index], pers.truth[:, :, index], inside), 2
            )
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("observed PM2.5 bin")
    frame.attrs["model"] = run.name
    frame.attrs["horizon"] = horizon
    return frame


def top_decile_table(runs_root: Path | str, names: Sequence[str] | None = None, horizon: int = 24) -> pd.DataFrame:
    """MAE and bias restricted to the highest 10 % of observed targets."""
    rows = []
    for run in list_runs(runs_root):
        if names is not None and run.name not in names:
            continue
        pred = _load_predictions(run)
        if pred is None:
            continue
        mask = top_decile_mask(pred)[:, :, horizon - 1]
        index = horizon - 1
        rows.append(
            {
                "model": run.name,
                "seed": run.seed,
                f"top-decile MAE h={horizon}": masked_mae(pred.pred[:, :, index], pred.truth[:, :, index], mask),
                f"top-decile bias h={horizon}": masked_bias(pred.pred[:, :, index], pred.truth[:, :, index], mask),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).groupby("model").mean(numeric_only=True).drop(columns="seed").round(2)
    order = [n for n in BASELINES + GRID if n in frame.index]
    return frame.loc[order + [n for n in frame.index if n not in order]]


def worst_weeks(
    dataset, runs_root: Path | str, model: str, horizon: int = 24, n: int = 5, seed: int | None = None
) -> pd.DataFrame:
    """The test weeks where the model was worst, with what was happening.

    Failure analysis starts from *when* it failed. Almost always the answer is
    an episode: a build-up the features could not anticipate, or a fireworks
    spike no feature set could.
    """
    run = _pick_run(runs_root, model, seed)
    if run is None:
        return pd.DataFrame()
    pred = _load_predictions(run)
    if pred is None:
        return pd.DataFrame()

    index = horizon - 1
    weeks = bootstrap.week_ids(pred.origins)
    rows = []
    for week in np.unique(weeks):
        rows_in = np.flatnonzero(weeks == week)
        p, t, m = pred.pred[rows_in, :, index], pred.truth[rows_in, :, index], pred.mask[rows_in, :, index]
        if not m.any():
            continue
        # Label by the first forecast *valid* hour (origin + 1), which is the
        # first hour the week's errors are actually measured at.
        first_valid = int(pred.origins[rows_in[0]]) + 1
        rows.append(
            {
                "week starting": str(dataset.timestamps[first_valid].astype("datetime64[D]")),
                "MAE": round(masked_mae(p, t, m), 2),
                "bias": round(masked_bias(p, t, m), 2),
                "mean observed": round(float(t[m].mean()), 1),
                "max observed": round(float(t[m].max()), 0),
            }
        )
    frame = pd.DataFrame(rows).set_index("week starting")
    frame.attrs["model"] = run.name
    return frame.sort_values("MAE", ascending=False).head(n)


def controls_table(runs_root: Path | str, model: str = "d1_wind_crossview") -> pd.DataFrame | None:
    """Negative-control results across every seed that has them, or ``None``.

    Positive damage means the perturbation made the forecast worse. The two
    reference rows are the scale; read every other row as a fraction of them.
    """
    rows = []
    for run in list_runs(runs_root):
        if run.name != model:
            continue
        path = run.path / "controls.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for record in payload.get("controls", []):
            for h, cell in record["horizons"].items():
                rows.append(
                    {
                        "control": record["control"],
                        "reference": bool(record.get("is_reference")),
                        "seed": run.seed,
                        "horizon": int(h),
                        "damage": cell["damage"],
                        "detected": bool(cell["detected"]),
                    }
                )
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["control", "horizon"])
        .agg(
            damage=("damage", "mean"),
            seed_sd=("damage", lambda s: s.std(ddof=1) if len(s) > 1 else 0.0),
            detected=("detected", "sum"),
            seeds=("seed", "nunique"),
            reference=("reference", "first"),
        )
        .reset_index()
    )
    summary["detected"] = summary["detected"].astype(int).astype(str) + "/" + summary["seeds"].astype(str)
    summary = summary.drop(columns="seeds")
    summary["damage"] = summary["damage"].round(3)
    summary["seed_sd"] = summary["seed_sd"].round(3)
    # References first, then the controls, so the scale is read before the rows
    # that depend on it.
    summary["_order"] = (~summary["reference"]).astype(int)
    summary = summary.sort_values(["_order", "control", "horizon"]).drop(columns="_order")
    return summary.set_index(["control", "horizon"])


# ------------------------------------------------------- 13. conclusions
def headline(runs_root: Path | str) -> pd.DataFrame:
    """The numbers the conclusion is written from, in one place."""
    numeric = metrics_numeric(runs_root, "mae")
    if numeric.empty:
        return pd.DataFrame()
    rows = []
    for name in ("persistence", "lightgbm", "t0_temporal_only", "d1_wind_crossview"):
        if name in numeric.index:
            rows.append({"model": name, **{c: round(float(v), 2) for c, v in numeric.loc[name].items()}})
    return pd.DataFrame(rows).set_index("model")

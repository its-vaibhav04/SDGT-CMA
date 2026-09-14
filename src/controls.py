"""Negative controls: perturb a trained model's inputs and check what moves.

    python scripts/run_controls.py --run experiments/runs/d1_wind_crossview_seed42

Run 2 produced a null result -- no spatial component beat a purely temporal
model -- and the components demonstrably trained while producing it
(``prior_strength`` 1.00 -> 0.61, the fusion gate developing real per-channel
structure). That combination is what these controls exist to interpret.

**The problem a control solves.** "Adding the wind graph did not improve MAE" is
a failure to find an effect, which is the weakest form of a negative claim. It is
equally consistent with a graph that is inert and a graph that helps somewhere
the metric does not look. A control answers a sharper question: *does this model
depend on the wind field at all?* If the wind can be reversed -- every transport
direction turned through 180 degrees -- and the forecast does not move, the graph
was never using wind, and no attention heatmap rescues the claim.

**Why a power reference is mandatory here.** The head is persistence-anchored:
``forecast = y_t + head(fused)``. The anchor is not perturbed by any control, so
the *most* any input perturbation can move is the learned correction sitting on
top of persistence. A control that reports "no change" is uninterpretable until
you know how much there was to change. ``zero_correction`` supplies that
denominator by zeroing the head, and ``history_shuffle`` supplies the matching
upper bound by destroying temporal order. Read every other control as a fraction
of those two, never in isolation.

**What isolates what.** ``wind_u`` and ``wind_v`` are model features *and*
``wind_uv`` is fed separately to the graph. Negating only ``wind_uv`` therefore
perturbs the graph while leaving the temporal branch's view of wind untouched --
which is the clean single-component test. ``wind_reversal_full`` negates both and
asks the whole-model question instead. The two are reported side by side because
their difference is itself informative.

Controls never train and never touch the training code. Each one re-runs the test
split through an unmodified checkpoint and returns a :class:`Predictions` that is
row-aligned with the unperturbed run, so ``src.bootstrap.paired_interval``
applies directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.data.contract import ProcessedDataset
from src.data.windows import Batch, WindowBatcher
from src.metrics import Predictions

# A perturbation of the batch handed to the model.
BatchFn = Callable[[Batch], Batch]
# A perturbation of (adjacency, lag) as the graph emits them.
GraphFn = Callable[[torch.Tensor, torch.Tensor], "tuple[torch.Tensor, torch.Tensor]"]


@dataclass(frozen=True)
class Control:
    """One perturbation, plus what a null result from it would mean."""

    name: str
    description: str
    reads: str
    """Which claim this control protects, in one phrase."""

    batch_fn: BatchFn | None = None
    graph_fn: GraphFn | None = None
    zero_head: bool = False

    requires_graph: bool = False
    """Graph perturbations are meaningless for the temporal-only configuration."""

    is_reference: bool = False
    """References calibrate the scale; they are not evidence of a dependency."""

    def applies_to(self, model: nn.Module) -> bool:
        return not self.requires_graph or getattr(model, "graph", None) is not None


# ------------------------------------------------------------ module swapping
class _PerturbedGraph(nn.Module):
    """Wraps a graph module and rewrites what it emits.

    The graph's own diagnostics are passed through untouched: they describe what
    the *unperturbed* graph computed, which is what makes a line like "the graph
    still reports a 2.91 h mean transport lag, and reversing the wind changed
    nothing" possible to write.
    """

    def __init__(self, inner: nn.Module, transform: GraphFn):
        super().__init__()
        self.inner = inner
        self.transform = transform

    def forward(self, wind_uv: torch.Tensor):
        adjacency, lag, diagnostics = self.inner(wind_uv)
        adjacency, lag = self.transform(adjacency, lag)
        return adjacency, lag, diagnostics


class _ZeroHead(nn.Module):
    """Returns zeros in the head's output shape, leaving only the anchor."""

    def __init__(self, inner: nn.Module, horizon: int):
        super().__init__()
        self.inner = inner
        self.horizon = horizon

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            fused.shape[0], fused.shape[1], self.horizon,
            device=fused.device, dtype=fused.dtype,
        )


@contextmanager
def _applied(model: nn.Module, control: Control | None) -> Iterator[None]:
    """Install a control's module swaps, and always take them back out.

    A control that leaked would silently contaminate every later control in the
    same process, and the failure would look like a result rather than a bug.
    ``None`` installs nothing, so the unperturbed baseline runs through this same
    path and cannot drift from the controls it is compared against.
    """
    if control is None:
        yield
        return

    original_graph = getattr(model, "graph", None)
    original_head = getattr(model, "head", None)
    try:
        if control.graph_fn is not None:
            if original_graph is None:
                raise ValueError(f"control {control.name!r} needs a graph, model has none")
            model.graph = _PerturbedGraph(original_graph, control.graph_fn)
        if control.zero_head:
            model.head = _ZeroHead(original_head, model.config.horizon)
        yield
    finally:
        if control.graph_fn is not None:
            model.graph = original_graph
        if control.zero_head:
            model.head = original_head


# ------------------------------------------------------------- perturbations
def derangement(n: int, seed: int = 0) -> np.ndarray:
    """A permutation with no fixed point.

    A plain shuffle would leave some stations mapped to themselves, weakening the
    control by exactly the fraction it failed to move. Rejection sampling is fine
    here: the probability a random permutation is a derangement tends to 1/e, so
    this succeeds in a couple of draws.
    """
    if n < 2:
        raise ValueError("a derangement needs at least two elements")
    rng = np.random.default_rng(seed)
    while True:
        candidate = rng.permutation(n)
        if not (candidate == np.arange(n)).any():
            return candidate


def _reverse_wind_uv(batch: Batch) -> Batch:
    return Batch(
        features=batch.features,
        wind_uv=-batch.wind_uv,
        target=batch.target,
        target_mask=batch.target_mask,
        origins=batch.origins,
        anchor=batch.anchor,
    )


def _reverse_wind_everywhere(dataset: ProcessedDataset) -> BatchFn:
    """Negate raw wind in the graph input *and* the standardised feature columns.

    The features are standardised, so negating the raw value is not the same as
    negating the stored one. With ``z = (x - mu) / sigma``, the column holding
    ``-x`` is ``-z - 2 * mu / sigma``; dropping that shift would quietly reverse
    the wind *and* move its mean, confounding the control with an offset.
    """
    specs = {spec.name: spec for spec in dataset.scaler.specs}
    shifts: dict[int, float] = {}
    for name in ("wind_u", "wind_v"):
        spec = specs[name]
        if spec.kind != "standard":
            raise ValueError(
                f"{name} is transformed as {spec.kind!r}; the reflection formula "
                "here assumes plain standardisation"
            )
        shifts[dataset.feature_index(name)] = 2.0 * spec.mean / spec.std

    def apply(batch: Batch) -> Batch:
        features = batch.features.clone()
        for index, shift in shifts.items():
            features[..., index] = -features[..., index] - shift
        return Batch(
            features=features,
            wind_uv=-batch.wind_uv,
            target=batch.target,
            target_mask=batch.target_mask,
            origins=batch.origins,
            anchor=batch.anchor,
        )

    return apply


def _shuffle_history(seed: int = 0) -> BatchFn:
    """Scramble the lookback hours, destroying temporal order.

    The positive control. The anchor is untouched, so this does not remove
    persistence -- it removes everything the model learned to add on top of it,
    which is precisely the quantity every other control is trying to move.
    """
    generator = torch.Generator().manual_seed(seed)

    def apply(batch: Batch) -> Batch:
        lookback = batch.features.shape[1]
        order = torch.randperm(lookback, generator=generator).to(batch.features.device)
        return Batch(
            features=batch.features[:, order],
            wind_uv=batch.wind_uv[:, order],
            target=batch.target,
            target_mask=batch.target_mask,
            origins=batch.origins,
            anchor=batch.anchor,
        )

    return apply


def _permute_edges(n_stations: int, seed: int = 0) -> GraphFn:
    """Relabel every edge's source station, keeping the weight distribution.

    Each target still receives the same number of edges carrying the same
    weights and the same travel times -- they simply arrive from the wrong
    stations. Row sums are preserved exactly, because permuting entries within a
    row cannot change their total, so the adjacency stays a distribution over
    sources and nothing downstream sees a malformed graph.
    """
    order = torch.from_numpy(derangement(n_stations, seed))

    def apply(adjacency: torch.Tensor, lag: torch.Tensor):
        index = order.to(adjacency.device)
        return adjacency[..., index], lag[..., index]

    return apply


def _identity_adjacency(n_stations: int) -> GraphFn:
    """Cut every cross-station edge, leaving each station reading only itself.

    Lag goes to zero rather than the graph's ``lag_min``: the condition being
    tested is "no spatial information at all", and a self-edge at a one-hour lag
    would still be a (degenerate) temporal intervention.
    """
    def apply(adjacency: torch.Tensor, lag: torch.Tensor):
        eye = torch.eye(n_stations, device=adjacency.device, dtype=adjacency.dtype)
        return eye.expand_as(adjacency).contiguous(), torch.zeros_like(lag)

    return apply


def build_controls(
    dataset: ProcessedDataset,
    n_stations: int,
    seed: int = 0,
) -> list[Control]:
    """The full suite, ordered so the report reads top to bottom as an argument."""
    return [
        Control(
            name="wind_reversal",
            description="negate the wind field the graph sees (u, v -> -u, -v)",
            reads="the graph's dependence on wind direction, in isolation",
            batch_fn=_reverse_wind_uv,
            requires_graph=True,
        ),
        Control(
            name="edge_permutation",
            description="relabel each edge's source station by a fixed derangement",
            reads="whether it matters *which* stations a target attends to",
            graph_fn=_permute_edges(n_stations, seed),
            requires_graph=True,
        ),
        Control(
            name="identity_adjacency",
            description="cut every cross-station edge; each station sees only itself",
            reads="the total contribution of cross-station information",
            graph_fn=_identity_adjacency(n_stations),
            requires_graph=True,
        ),
        Control(
            name="wind_reversal_full",
            description="negate wind in the graph input and the feature columns",
            reads="the whole model's dependence on wind, graph and temporal branch",
            batch_fn=_reverse_wind_everywhere(dataset),
        ),
        Control(
            name="history_shuffle",
            description="scramble the lookback hours, destroying temporal order",
            reads="POWER CHECK: the largest move any input perturbation can make",
            batch_fn=_shuffle_history(seed),
            is_reference=True,
        ),
        Control(
            name="zero_correction",
            description="zero the head, leaving the persistence anchor alone",
            reads="DENOMINATOR: what the learned correction is worth in total",
            zero_head=True,
            is_reference=True,
        ),
    ]


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate_control(
    model: nn.Module,
    batcher: WindowBatcher,
    dataset: ProcessedDataset,
    control: Control | None,
    *,
    device: torch.device,
    model_name: str,
    seed: int,
) -> Predictions:
    """Score the test split once, under one control (or none, for the baseline).

    Returns predictions in ug/m3, row-aligned with every other call on the same
    batcher -- the control changes what the model reads, never which windows are
    scored or which targets are observed, so the pairing is exact by
    construction.
    """
    model.eval()
    predictions, truths, masks, origins = [], [], [], []

    name = "unperturbed" if control is None else control.name
    with _applied(model, control):
        for batch in batcher:
            batch = batch.to(device)
            if control is not None and control.batch_fn is not None:
                batch = control.batch_fn(batch)
            output, _ = model(batch.features, batch.wind_uv, batch.anchor)

            predictions.append(output.cpu().numpy())
            truths.append(batch.target.cpu().numpy())
            masks.append(batch.target_mask.cpu().numpy())
            origins.append(batch.origins.cpu().numpy())

    return Predictions(
        origins=np.concatenate(origins),
        pred=dataset.target_scaler.inverse(np.concatenate(predictions)).astype(np.float32),
        truth=dataset.target_scaler.inverse(np.concatenate(truths)).astype(np.float32),
        mask=np.concatenate(masks).astype(bool),
        model=f"{model_name}::{name}",
        seed=seed,
    )


@torch.no_grad()
def station_influence(
    model: nn.Module,
    batcher: WindowBatcher,
    dataset: ProcessedDataset,
    *,
    device: torch.device,
    horizons: Sequence[int] = (1, 6, 12, 24),
) -> list[dict[str, float]]:
    """Occlude one station's inputs at a time; measure the damage to the others.

    Occluded features are set to zero, which is the training-split mean in
    standardised space -- the least informative value the column can take, rather
    than an out-of-distribution one that would test extrapolation instead of
    dependence.

    **The occluded station is excluded from its own scoring.** Its forecast
    collapses towards the persistence anchor whatever the graph does, so
    including it would measure the anchor and hide the quantity of interest:
    *how much station j's observations are worth to everybody else.* The
    unperturbed reference is scored on that same reduced mask, so each delta
    reflects the occlusion alone and not a change in which rows were counted.
    """
    from src.metrics import masked_mae

    def collect(occluded: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds, truths, masks = [], [], []
        for batch in batcher:
            batch = batch.to(device)
            features, wind_uv = batch.features, batch.wind_uv
            if occluded is not None:
                features = features.clone()
                features[:, :, occluded, :] = 0.0
                wind_uv = wind_uv.clone()
                wind_uv[:, :, occluded, :] = 0.0
            output, _ = model(features, wind_uv, batch.anchor)
            preds.append(output.cpu().numpy())
            truths.append(batch.target.cpu().numpy())
            masks.append(batch.target_mask.cpu().numpy())
        return (
            dataset.target_scaler.inverse(np.concatenate(preds)),
            dataset.target_scaler.inverse(np.concatenate(truths)),
            np.concatenate(masks).astype(bool),
        )

    def score(pred, truth, mask, excluded: int) -> dict[int, float]:
        reduced = mask.copy()
        reduced[:, excluded, :] = False
        return {
            h: masked_mae(pred[:, :, h - 1], truth[:, :, h - 1], reduced[:, :, h - 1])
            for h in horizons
        }

    model.eval()
    # The unperturbed pass is computed once and re-scored under each station's
    # mask; recomputing it per station would be N-1 wasted forward passes.
    base_pred, truth, mask = collect(None)

    rows: list[dict[str, float]] = []
    for index in range(dataset.n_stations):
        occluded_pred, _, _ = collect(index)
        occluded_scores = score(occluded_pred, truth, mask, index)
        baseline_scores = score(base_pred, truth, mask, index)
        rows.append(
            {
                "station": dataset.stations[index],
                **{f"delta_h{h}": occluded_scores[h] - baseline_scores[h] for h in horizons},
                **{f"mae_h{h}": occluded_scores[h] for h in horizons},
            }
        )
    return rows

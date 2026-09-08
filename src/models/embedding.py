"""Feature embedding shared by every model configuration.

Projects the raw feature vector to the model width and adds a learned station
identity. The station embedding is not decoration: the temporal encoder shares
one set of weights across all stations, so without it the model literally cannot
tell Huairou (a clean northern suburb, mean 62 ug/m3) from Dongsi (central,
mean 85). The graph knows about station identity through its geometry; the
temporal branch would not.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureEmbedding(nn.Module):
    """``[B, L, N, F] -> [B, L, N, d]``."""

    def __init__(
        self,
        n_features: int,
        n_stations: int,
        d_model: int,
        dropout: float = 0.0,
        use_station_embedding: bool = True,
    ):
        super().__init__()
        self.projection = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_station_embedding = use_station_embedding
        if use_station_embedding:
            self.station = nn.Parameter(torch.zeros(n_stations, d_model))
            nn.init.normal_(self.station, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.projection(features)
        if self.use_station_embedding:
            hidden = hidden + self.station          # broadcasts over B and L
        return self.dropout(hidden)

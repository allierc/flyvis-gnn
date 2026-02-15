"""Neuron state dataclasses for flyvis simulation.

Replaces the packed (N, 9) tensor with named fields.
Follows the zapbench pattern: data loads directly into dataclass fields,
classmethods handle I/O, no raw tensor layout leaks outside.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class NeuronState:
    """Single-frame neuron state.

    Static fields (set once, never change per frame):
        index, pos, group_type, neuron_type

    Dynamic fields (updated every simulation frame):
        voltage, stimulus, calcium, fluorescence
    """

    # static
    index: torch.Tensor        # (N,) long — neuron IDs 0..N-1
    pos: torch.Tensor          # (N, 2) float32 — spatial (x, y)
    group_type: torch.Tensor   # (N,) long — grouped neuron type
    neuron_type: torch.Tensor  # (N,) long — integer neuron type

    # dynamic
    voltage: torch.Tensor      # (N,) float32 — membrane voltage u
    stimulus: torch.Tensor     # (N,) float32 — visual input / excitation
    calcium: torch.Tensor      # (N,) float32 — calcium concentration
    fluorescence: torch.Tensor # (N,) float32 — fluorescence readout

    @property
    def n_neurons(self) -> int:
        return self.index.shape[0]

    @property
    def device(self) -> torch.device:
        return self.voltage.device

    def observable(self, calcium_type: str = "none") -> torch.Tensor:
        """Return the observable signal: voltage or calcium, as (N, 1)."""
        if calcium_type != "none":
            return self.calcium.unsqueeze(-1)
        return self.voltage.unsqueeze(-1)

    @classmethod
    def from_numpy(cls, x: np.ndarray) -> NeuronState:
        """Create from legacy (N, 9) numpy array.

        Column layout: [index, xpos, ypos, voltage, stimulus,
                        group_type, neuron_type, calcium, fluorescence]
        """
        t = torch.from_numpy(x) if not isinstance(x, torch.Tensor) else x
        return cls(
            index=t[:, 0].long(),
            pos=t[:, 1:3].float(),
            group_type=t[:, 5].long(),
            neuron_type=t[:, 6].long(),
            voltage=t[:, 3].float(),
            stimulus=t[:, 4].float(),
            calcium=t[:, 7].float(),
            fluorescence=t[:, 8].float(),
        )

    def to_packed(self) -> torch.Tensor:
        """Pack back into (N, 9) tensor for legacy compatibility."""
        x = torch.zeros(self.n_neurons, 9, dtype=torch.float32, device=self.device)
        x[:, 0] = self.index.float()
        x[:, 1:3] = self.pos
        x[:, 3] = self.voltage
        x[:, 4] = self.stimulus
        x[:, 5] = self.group_type.float()
        x[:, 6] = self.neuron_type.float()
        x[:, 7] = self.calcium
        x[:, 8] = self.fluorescence
        return x

    def to(self, device: torch.device) -> NeuronState:
        """Move all tensors to device."""
        return NeuronState(
            index=self.index.to(device),
            pos=self.pos.to(device),
            group_type=self.group_type.to(device),
            neuron_type=self.neuron_type.to(device),
            voltage=self.voltage.to(device),
            stimulus=self.stimulus.to(device),
            calcium=self.calcium.to(device),
            fluorescence=self.fluorescence.to(device),
        )


@dataclass
class NeuronTimeSeries:
    """Full simulation timeseries — static metadata + dynamic per-frame data.

    Static fields are stored once (same for all frames).
    Dynamic fields have a leading time dimension (T, N).
    """

    # static (stored once)
    index: torch.Tensor        # (N,)
    pos: torch.Tensor          # (N, 2)
    group_type: torch.Tensor   # (N,)
    neuron_type: torch.Tensor  # (N,)

    # dynamic (stored per frame)
    voltage: torch.Tensor      # (T, N)
    stimulus: torch.Tensor     # (T, N)
    calcium: torch.Tensor      # (T, N)
    fluorescence: torch.Tensor # (T, N)

    @property
    def n_frames(self) -> int:
        return self.voltage.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.index.shape[0]

    def frame(self, t: int) -> NeuronState:
        """Extract single-frame NeuronState at time t."""
        return NeuronState(
            index=self.index,
            pos=self.pos,
            group_type=self.group_type,
            neuron_type=self.neuron_type,
            voltage=self.voltage[t],
            stimulus=self.stimulus[t],
            calcium=self.calcium[t],
            fluorescence=self.fluorescence[t],
        )

    def subset_neurons(self, ids: np.ndarray | torch.Tensor) -> NeuronTimeSeries:
        """Select a subset of neurons by index."""
        return NeuronTimeSeries(
            index=self.index[ids],
            pos=self.pos[ids],
            group_type=self.group_type[ids],
            neuron_type=self.neuron_type[ids],
            voltage=self.voltage[:, ids],
            stimulus=self.stimulus[:, ids],
            calcium=self.calcium[:, ids],
            fluorescence=self.fluorescence[:, ids],
        )

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> NeuronTimeSeries:
        """Create from legacy (T, N, 9) numpy array.

        Column layout: [index, xpos, ypos, voltage, stimulus,
                        group_type, neuron_type, calcium, fluorescence]
        """
        t = torch.from_numpy(arr) if not isinstance(arr, torch.Tensor) else arr
        return cls(
            # static — take from first frame
            index=t[0, :, 0].long(),
            pos=t[0, :, 1:3].float(),
            group_type=t[0, :, 5].long(),
            neuron_type=t[0, :, 6].long(),
            # dynamic — all frames
            voltage=t[:, :, 3].float(),
            stimulus=t[:, :, 4].float(),
            calcium=t[:, :, 7].float(),
            fluorescence=t[:, :, 8].float(),
        )

    @classmethod
    def from_zarr_v2(cls, path: str | Path) -> NeuronTimeSeries:
        """Load from V2 zarr split format.

        Expects:
            path/metadata.zarr    — (N, 5) static: [index, xpos, ypos, group_type, type]
            path/timeseries.zarr  — (T, N, 4) dynamic: [voltage, stimulus, calcium, fluorescence]
        """
        import tensorstore as ts

        path = Path(path)
        metadata_path = path / 'metadata.zarr'
        timeseries_path = path / 'timeseries.zarr'

        meta_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(metadata_path)},
        }
        metadata = ts.open(meta_spec).result().read().result()  # (N, 5)

        ts_spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(timeseries_path)},
        }
        timeseries = ts.open(ts_spec).result().read().result()  # (T, N, 4)

        return cls(
            index=torch.from_numpy(metadata[:, 0].copy()).long(),
            pos=torch.from_numpy(metadata[:, 1:3].copy()).float(),
            group_type=torch.from_numpy(metadata[:, 3].copy()).long(),
            neuron_type=torch.from_numpy(metadata[:, 4].copy()).long(),
            voltage=torch.from_numpy(timeseries[:, :, 0].copy()).float(),
            stimulus=torch.from_numpy(timeseries[:, :, 1].copy()).float(),
            calcium=torch.from_numpy(timeseries[:, :, 2].copy()).float(),
            fluorescence=torch.from_numpy(timeseries[:, :, 3].copy()).float(),
        )

    @classmethod
    def from_zarr_v1(cls, path: str | Path) -> NeuronTimeSeries:
        """Load from V1 zarr format (single array, T x N x 9)."""
        import tensorstore as ts

        zarr_path = Path(str(path) + '.zarr') if not str(path).endswith('.zarr') else Path(path)
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(zarr_path)},
        }
        arr = ts.open(spec).result().read().result()  # (T, N, 9)
        return cls.from_numpy(arr)

    @classmethod
    def load(cls, path: str | Path) -> NeuronTimeSeries:
        """Auto-detect format (zarr_v2, zarr_v1, npy) and load.

        Checks for:
            1. V2 zarr directory (path/ with metadata.zarr + timeseries.zarr)
            2. V1 zarr file (path.zarr with .zarray)
            3. NumPy file (path.npy)
        """
        path = Path(path)
        base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

        # check V2 zarr
        if base_path.exists() and base_path.is_dir():
            if (base_path / 'metadata.zarr').exists() and (base_path / 'timeseries.zarr').exists():
                return cls.from_zarr_v2(base_path)

        # check V1 zarr
        zarr_v1_path = Path(str(base_path) + '.zarr')
        if zarr_v1_path.exists() and zarr_v1_path.is_dir():
            if (zarr_v1_path / '.zarray').exists():
                return cls.from_zarr_v1(base_path)

        # check npy
        npy_path = Path(str(base_path) + '.npy')
        if npy_path.exists():
            return cls.from_numpy(np.load(npy_path))

        raise FileNotFoundError(f"no .npy or .zarr found at {base_path}")

    def to_packed(self) -> np.ndarray:
        """Convert to legacy (T, N, 9) numpy array."""
        T, N = self.n_frames, self.n_neurons
        full = np.empty((T, N, 9), dtype=np.float32)

        # static — broadcast from (N,) to (T, N)
        full[:, :, 0] = self.index.numpy()
        full[:, :, 1:3] = self.pos.numpy()
        full[:, :, 5] = self.group_type.numpy()
        full[:, :, 6] = self.neuron_type.numpy()

        # dynamic
        full[:, :, 3] = self.voltage.numpy()
        full[:, :, 4] = self.stimulus.numpy()
        full[:, :, 7] = self.calcium.numpy()
        full[:, :, 8] = self.fluorescence.numpy()

        return full

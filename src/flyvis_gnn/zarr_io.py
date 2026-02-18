"""zarr/tensorstore I/O utilities for simulation data.

provides:
- ZarrSimulationWriter: incremental writer (legacy V1 format)
- ZarrSimulationWriterV2: split metadata/timeseries writer (V2 format)
- ZarrSimulationWriterV3: per-field writer (V3 format — primary)
- detect_format: check if V3 zarr or .npy exists at path
- load_simulation_data: load as NeuronTimeSeries with optional field selection
- load_raw_array: load raw numpy array from zarr or npy (for derivative targets etc.)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tensorstore as ts

if TYPE_CHECKING:
    from flyvis_gnn.neuron_state import NeuronState, NeuronTimeSeries

# V2 writer internals: static and dynamic column layout
_STATIC_COLS = [0, 1, 2, 5, 6]   # INDEX, XPOS, YPOS, GROUP_TYPE, TYPE
_DYNAMIC_COLS = [3, 4, 7, 8]     # VOLTAGE, STIMULUS, CALCIUM, FLUORESCENCE
_N_DYNAMIC = len(_DYNAMIC_COLS)


class ZarrSimulationWriter:
    """incremental writer - appends frames during generation (legacy V1 format).

    usage:
        writer = ZarrSimulationWriter(path, n_neurons=1000, n_features=8)
        for frame in simulation:
            writer.append(frame)
        writer.finalize()
    """

    def __init__(
        self,
        path: str | Path,
        n_neurons: int,
        n_features: int,
        chunks: tuple[int, int, int] | None = None,
        dtype: np.dtype = np.float32,
    ):
        """initialize zarr writer.

        args:
            path: output path (without extension, .zarr will be added)
            n_neurons: number of neurons (second dimension)
            n_features: number of features per neuron (third dimension)
            chunks: chunk sizes (time, neurons, features). use -1 for full dimension.
            dtype: data type for storage
        """
        self.path = Path(path)
        if not str(self.path).endswith('.zarr'):
            self.path = Path(str(self.path) + '.zarr')

        self.n_neurons = n_neurons
        self.n_features = n_features
        self.dtype = dtype

        # determine chunk sizes
        if chunks is None:
            # default: 500 frames, full neurons, full features
            chunks = (500, n_neurons, n_features)
        else:
            # replace -1 with actual dimensions
            chunks = (
                chunks[0],
                n_neurons if chunks[1] == -1 else chunks[1],
                n_features if chunks[2] == -1 else chunks[2],
            )
        self.chunks = chunks

        # buffer for accumulating frames before writing
        self._buffer: list[np.ndarray] = []
        self._total_frames = 0
        self._store: ts.TensorStore | None = None
        self._initialized = False

    def _initialize_store(self, first_frame: np.ndarray):
        """initialize tensorstore with zarr format."""
        # ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # remove existing zarr directory if present (ignore_errors for NFS)
        if self.path.exists():
            import shutil
            shutil.rmtree(self.path, ignore_errors=True)

        # create zarr store with tensorstore
        # start with initial capacity, will resize as needed
        initial_time_capacity = max(self.chunks[0] * 10, 1000)

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(self.path),
            },
            'metadata': {
                'dtype': '<f4' if self.dtype == np.float32 else '<f8',
                'shape': [initial_time_capacity, self.n_neurons, self.n_features],
                'chunks': list(self.chunks),
                'compressor': {
                    'id': 'blosc',
                    'cname': 'zstd',
                    'clevel': 3,
                    'shuffle': 2,  # bitshuffle
                },
            },
            'create': True,
            'delete_existing': True,
        }

        self._store = ts.open(spec).result()
        self._initialized = True

    def append(self, frame: np.ndarray):
        """append a single frame to the buffer.

        args:
            frame: array of shape (n_neurons, n_features)
        """
        if frame.shape != (self.n_neurons, self.n_features):
            raise ValueError(
                f"frame shape {frame.shape} doesn't match expected "
                f"({self.n_neurons}, {self.n_features})"
            )

        self._buffer.append(frame.astype(self.dtype, copy=False))

        # flush buffer when it reaches chunk size
        if len(self._buffer) >= self.chunks[0]:
            self._flush_buffer()

    def _flush_buffer(self):
        """write buffered frames to zarr store."""
        if not self._buffer:
            return

        # stack frames into array
        data = np.stack(self._buffer, axis=0)
        n_frames = data.shape[0]

        if not self._initialized:
            self._initialize_store(self._buffer[0])

        # check if we need to resize
        current_shape = self._store.shape
        needed_size = self._total_frames + n_frames
        if needed_size > current_shape[0]:
            # resize to accommodate new data (with some headroom)
            new_size = max(needed_size, current_shape[0] * 2)
            self._store = self._store.resize(
                exclusive_max=[new_size, self.n_neurons, self.n_features]
            ).result()

        # write data
        self._store[self._total_frames:self._total_frames + n_frames].write(data).result()

        self._total_frames += n_frames
        self._buffer.clear()

    def finalize(self):
        """finalize the zarr store - flush remaining buffer and resize to exact size."""
        # flush any remaining buffered data
        self._flush_buffer()

        if self._store is not None and self._total_frames > 0:
            # resize to exact final size
            self._store = self._store.resize(
                exclusive_max=[self._total_frames, self.n_neurons, self.n_features]
            ).result()

        return self._total_frames


class ZarrSimulationWriterV2:
    """split metadata/timeseries writer for efficient storage (V2 format).

    separates static columns (INDEX, XPOS, YPOS, GROUP_TYPE, TYPE) from
    dynamic columns (VOLTAGE, STIMULUS, CALCIUM, FLUORESCENCE) to avoid
    redundant storage of position data.

    storage structure:
        path/
            metadata.zarr    # (N, 5) static columns, stored once
            timeseries.zarr  # (T, N, 4) dynamic columns

    usage:
        writer = ZarrSimulationWriterV2(path, n_neurons=14011, n_features=9)
        for frame in simulation:
            writer.append(frame)  # frame is (N, 9)
        writer.finalize()
    """

    def __init__(
        self,
        path: str | Path,
        n_neurons: int,
        n_features: int = 9,
        time_chunks: int = 2000,
        dtype: np.dtype = np.float32,
    ):
        """initialize V2 zarr writer.

        args:
            path: output directory path
            n_neurons: number of neurons
            n_features: total features per neuron (must be 9 for flyvis)
            time_chunks: chunk size along time dimension for timeseries
            dtype: data type for storage
        """
        self.path = Path(path)
        self.n_neurons = n_neurons
        self.n_features = n_features
        self.time_chunks = time_chunks
        self.dtype = dtype

        # paths for sub-arrays
        self.metadata_path = self.path / 'metadata.zarr'
        self.timeseries_path = self.path / 'timeseries.zarr'

        # state
        self._metadata_saved = False
        self._metadata: np.ndarray | None = None
        self._buffer: list[np.ndarray] = []
        self._total_frames = 0
        self._ts_store: ts.TensorStore | None = None
        self._ts_initialized = False

    def _save_metadata(self, frame: np.ndarray):
        """save static metadata from first frame."""
        # ensure directory exists
        self.path.mkdir(parents=True, exist_ok=True)

        # remove existing if present (ignore_errors for NFS race conditions)
        if self.metadata_path.exists():
            import shutil
            shutil.rmtree(self.metadata_path, ignore_errors=True)

        # extract static columns: [INDEX, XPOS, YPOS, GROUP_TYPE, TYPE]
        self._metadata = frame[:, _STATIC_COLS].astype(self.dtype)  # (N, 5)

        # save metadata as zarr (small, no need for chunking)
        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(self.metadata_path),
            },
            'metadata': {
                'dtype': '<f4' if self.dtype == np.float32 else '<f8',
                'shape': list(self._metadata.shape),
                'chunks': list(self._metadata.shape),  # single chunk
                'compressor': {
                    'id': 'blosc',
                    'cname': 'zstd',
                    'clevel': 3,
                    'shuffle': 2,
                },
            },
            'create': True,
            'delete_existing': True,
        }

        store = ts.open(spec).result()
        store.write(self._metadata).result()
        self._metadata_saved = True

    def _initialize_timeseries_store(self):
        """initialize timeseries zarr store."""
        if self.timeseries_path.exists():
            import shutil
            shutil.rmtree(self.timeseries_path, ignore_errors=True)

        initial_time_capacity = max(self.time_chunks * 10, 1000)

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(self.timeseries_path),
            },
            'metadata': {
                'dtype': '<f4' if self.dtype == np.float32 else '<f8',
                'shape': [initial_time_capacity, self.n_neurons, _N_DYNAMIC],
                'chunks': [self.time_chunks, self.n_neurons, 1],  # column-chunked
                'compressor': {
                    'id': 'blosc',
                    'cname': 'zstd',
                    'clevel': 3,
                    'shuffle': 2,
                },
            },
            'create': True,
            'delete_existing': True,
        }

        self._ts_store = ts.open(spec).result()
        self._ts_initialized = True

    def append(self, frame: np.ndarray):
        """append a single frame from packed (N, 9) array.

        args:
            frame: array of shape (n_neurons, n_features) with all 9 columns
        """
        if frame.shape != (self.n_neurons, self.n_features):
            raise ValueError(
                f"frame shape {frame.shape} doesn't match expected "
                f"({self.n_neurons}, {self.n_features})"
            )

        # save metadata from first frame
        if not self._metadata_saved:
            self._save_metadata(frame)

        # extract dynamic columns: [VOLTAGE, STIMULUS, CALCIUM, FLUORESCENCE]
        dynamic_data = frame[:, _DYNAMIC_COLS].astype(self.dtype)
        self._buffer.append(dynamic_data)

        # flush when buffer reaches chunk size
        if len(self._buffer) >= self.time_chunks:
            self._flush_buffer()

    def append_state(self, state: NeuronState):
        """append a single frame from NeuronState dataclass.

        On first call, saves static metadata (index, pos, group_type, neuron_type).
        On every call, buffers dynamic fields (voltage, stimulus, calcium, fluorescence).
        """
        from flyvis_gnn.neuron_state import NeuronState as _NS
        from flyvis_gnn.utils import to_numpy

        if not self._metadata_saved:
            # build metadata: (N, 5) = [index, xpos, ypos, group_type, neuron_type]
            meta = np.column_stack([
                to_numpy(state.index.float()),
                to_numpy(state.pos),
                to_numpy(state.group_type.float()),
                to_numpy(state.neuron_type.float()),
            ]).astype(self.dtype)
            self._save_metadata_array(meta)

        # build dynamic: (N, 4) = [voltage, stimulus, calcium, fluorescence]
        dynamic = np.column_stack([
            to_numpy(state.voltage),
            to_numpy(state.stimulus),
            to_numpy(state.calcium),
            to_numpy(state.fluorescence),
        ]).astype(self.dtype)
        self._buffer.append(dynamic)

        if len(self._buffer) >= self.time_chunks:
            self._flush_buffer()

    def _save_metadata_array(self, metadata: np.ndarray):
        """save pre-built metadata array (N, 5)."""
        self.path.mkdir(parents=True, exist_ok=True)

        # ignore_errors for NFS race conditions on cluster filesystems
        if self.metadata_path.exists():
            import shutil
            shutil.rmtree(self.metadata_path, ignore_errors=True)

        self._metadata = metadata

        spec = {
            'driver': 'zarr',
            'kvstore': {
                'driver': 'file',
                'path': str(self.metadata_path),
            },
            'metadata': {
                'dtype': '<f4' if self.dtype == np.float32 else '<f8',
                'shape': list(self._metadata.shape),
                'chunks': list(self._metadata.shape),
                'compressor': {
                    'id': 'blosc',
                    'cname': 'zstd',
                    'clevel': 3,
                    'shuffle': 2,
                },
            },
            'create': True,
            'delete_existing': True,
        }

        store = ts.open(spec).result()
        store.write(self._metadata).result()
        self._metadata_saved = True

    def _flush_buffer(self):
        """write buffered timeseries data."""
        if not self._buffer:
            return

        data = np.stack(self._buffer, axis=0)  # (chunk_size, N, 4)
        n_frames = data.shape[0]

        if not self._ts_initialized:
            self._initialize_timeseries_store()

        # resize if needed
        current_shape = self._ts_store.shape
        needed_size = self._total_frames + n_frames
        if needed_size > current_shape[0]:
            new_size = max(needed_size, current_shape[0] * 2)
            self._ts_store = self._ts_store.resize(
                exclusive_max=[new_size, self.n_neurons, _N_DYNAMIC]
            ).result()

        # write
        self._ts_store[self._total_frames:self._total_frames + n_frames].write(data).result()

        self._total_frames += n_frames
        self._buffer.clear()

    def finalize(self):
        """finalize - flush buffer and resize to exact size."""
        self._flush_buffer()

        if self._ts_store is not None and self._total_frames > 0:
            self._ts_store = self._ts_store.resize(
                exclusive_max=[self._total_frames, self.n_neurons, _N_DYNAMIC]
            ).result()

        return self._total_frames


_DYNAMIC_FIELDS_V3 = ['voltage', 'stimulus', 'calcium', 'fluorescence']
_STATIC_FIELDS_V3 = ['index', 'pos', 'group_type', 'neuron_type']


class ZarrSimulationWriterV3:
    """Per-field zarr writer — each NeuronState key gets its own zarr array.

    Storage structure:
        path/
            index.zarr        # (N,) int32 — static
            pos.zarr          # (N, 2) float32 — static
            group_type.zarr   # (N,) int32 — static
            neuron_type.zarr  # (N,) int32 — static
            voltage.zarr      # (T, N) float32 — dynamic
            stimulus.zarr     # (T, N) float32 — dynamic
            calcium.zarr      # (T, N) float32 — dynamic
            fluorescence.zarr # (T, N) float32 — dynamic

    usage:
        writer = ZarrSimulationWriterV3(path, n_neurons=14011)
        for state in simulation:
            writer.append_state(state)
        writer.finalize()
    """

    def __init__(
        self,
        path: str | Path,
        n_neurons: int,
        time_chunks: int = 2000,
    ):
        self.path = Path(path)
        self.n_neurons = n_neurons
        self.time_chunks = time_chunks

        self._static_saved = False
        self._buffers: dict[str, list[np.ndarray]] = {f: [] for f in _DYNAMIC_FIELDS_V3}
        self._stores: dict[str, ts.TensorStore] = {}
        self._total_frames = 0
        self._dynamic_initialized = False

    def _save_static(self, state: NeuronState):
        """Save static fields from first NeuronState frame."""
        from flyvis_gnn.utils import to_numpy

        self.path.mkdir(parents=True, exist_ok=True)

        static_data = {
            'index': to_numpy(state.index).astype(np.int32),
            'pos': to_numpy(state.pos).astype(np.float32),
            'group_type': to_numpy(state.group_type).astype(np.int32),
            'neuron_type': to_numpy(state.neuron_type).astype(np.int32),
        }

        for name, data in static_data.items():
            zarr_path = self.path / f'{name}.zarr'
            if zarr_path.exists():
                import shutil
                shutil.rmtree(zarr_path, ignore_errors=True)

            dtype_str = '<i4' if data.dtype in (np.int32, np.int64) else '<f4'
            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': str(zarr_path)},
                'metadata': {
                    'dtype': dtype_str,
                    'shape': list(data.shape),
                    'chunks': list(data.shape),
                    'compressor': {
                        'id': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 2,
                    },
                },
                'create': True,
                'delete_existing': True,
            }
            store = ts.open(spec).result()
            store.write(data).result()

        self._static_saved = True

    def _initialize_dynamic_stores(self):
        """Create zarr stores for dynamic fields."""
        initial_cap = max(self.time_chunks * 10, 1000)

        for name in _DYNAMIC_FIELDS_V3:
            zarr_path = self.path / f'{name}.zarr'
            if zarr_path.exists():
                import shutil
                shutil.rmtree(zarr_path, ignore_errors=True)

            spec = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': str(zarr_path)},
                'metadata': {
                    'dtype': '<f4',
                    'shape': [initial_cap, self.n_neurons],
                    'chunks': [self.time_chunks, self.n_neurons],
                    'compressor': {
                        'id': 'blosc', 'cname': 'zstd', 'clevel': 3, 'shuffle': 2,
                    },
                },
                'create': True,
                'delete_existing': True,
            }
            self._stores[name] = ts.open(spec).result()

        self._dynamic_initialized = True

    def append_state(self, state: NeuronState):
        """Append one frame from NeuronState."""
        from flyvis_gnn.utils import to_numpy

        if not self._static_saved:
            self._save_static(state)

        self._buffers['voltage'].append(to_numpy(state.voltage).astype(np.float32))
        self._buffers['stimulus'].append(to_numpy(state.stimulus).astype(np.float32))
        self._buffers['calcium'].append(to_numpy(state.calcium).astype(np.float32))
        self._buffers['fluorescence'].append(to_numpy(state.fluorescence).astype(np.float32))

        if len(self._buffers['voltage']) >= self.time_chunks:
            self._flush_buffer()

    def _flush_buffer(self):
        """Write buffered dynamic data to zarr stores."""
        if not self._buffers['voltage']:
            return

        if not self._dynamic_initialized:
            self._initialize_dynamic_stores()

        n_frames = len(self._buffers['voltage'])

        for name in _DYNAMIC_FIELDS_V3:
            data = np.stack(self._buffers[name], axis=0)  # (chunk, N)

            # resize if needed
            current_shape = self._stores[name].shape
            needed = self._total_frames + n_frames
            if needed > current_shape[0]:
                new_size = max(needed, current_shape[0] * 2)
                self._stores[name] = self._stores[name].resize(
                    exclusive_max=[new_size, self.n_neurons]
                ).result()

            self._stores[name][self._total_frames:self._total_frames + n_frames].write(data).result()
            self._buffers[name].clear()

        self._total_frames += n_frames

    def finalize(self):
        """Flush remaining buffer and resize stores to exact size."""
        self._flush_buffer()

        for name in _DYNAMIC_FIELDS_V3:
            if name in self._stores and self._total_frames > 0:
                self._stores[name] = self._stores[name].resize(
                    exclusive_max=[self._total_frames, self.n_neurons]
                ).result()

        return self._total_frames


def detect_format(path: str | Path) -> Literal['npy', 'zarr_v3', 'none']:
    """check what format exists at path.

    args:
        path: base path without extension

    returns:
        'zarr_v3' if V3 zarr directory exists (per-field .zarr arrays)
        'npy' if .npy file exists
        'none' if nothing exists
    """
    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    # check for V3 zarr format (directory with per-field .zarr arrays)
    if base_path.exists() and base_path.is_dir():
        if (base_path / 'voltage.zarr').exists():
            return 'zarr_v3'

    # check for npy
    npy_path = Path(str(base_path) + '.npy')
    if npy_path.exists():
        return 'npy'

    return 'none'


def load_simulation_data(path: str | Path, fields=None) -> NeuronTimeSeries:
    """load simulation data as NeuronTimeSeries (V3 zarr or npy).

    args:
        path: base path (with or without extension)
        fields: list of field names to load (V3 only, e.g. ['voltage', 'stimulus']).
                None = all fields.

    returns:
        NeuronTimeSeries with requested fields (others are None)

    raises:
        FileNotFoundError: if no data found at path
    """
    from flyvis_gnn.neuron_state import NeuronTimeSeries
    return NeuronTimeSeries.load(path, fields=fields)


def load_raw_array(path: str | Path) -> np.ndarray:
    """load a raw numpy array from .zarr or .npy (for y_list derivative targets etc.).

    args:
        path: base path (with or without extension)

    returns:
        numpy array

    raises:
        FileNotFoundError: if no data found at path
    """
    path = Path(path)
    base_path = path.with_suffix('') if path.suffix in ('.npy', '.zarr') else path

    # try zarr (V1-style single array)
    zarr_path = Path(str(base_path) + '.zarr')
    if zarr_path.exists() and zarr_path.is_dir():
        spec = {
            'driver': 'zarr',
            'kvstore': {'driver': 'file', 'path': str(zarr_path)},
        }
        return ts.open(spec).result().read().result()

    # try npy
    npy_path = Path(str(base_path) + '.npy')
    if npy_path.exists():
        return np.load(npy_path)

    raise FileNotFoundError(f"no .zarr or .npy found at {base_path}")


def load_zarr_lazy(path: str | Path) -> ts.TensorStore:
    """load zarr file as tensorstore handle for lazy access.

    args:
        path: path to zarr directory

    returns:
        tensorstore handle
    """
    path = Path(path)
    if not str(path).endswith('.zarr'):
        path = Path(str(path) + '.zarr')

    spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': str(path),
        },
    }

    return ts.open(spec).result()

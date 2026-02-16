# Code Review & Engineering Roadmap

## Five Graph-Learning Repositories: NeuralGraph, ParticleGraph, MPM_pytorch, MetabolismGraph, flyvis-gnn

**Author**: Claude (Opus 4.6) | **Date**: 2026-02-14 | **Scope**: Architecture, code quality, refactoring guidelines, unification strategy, LLM compatibility

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [The 10 Engineering Issues](#2-the-10-engineering-issues)
3. [Core Architecture: Two Master-Level Components](#3-core-architecture)
4. [Refactoring Guidelines](#4-refactoring-guidelines)
5. [New Repository Development Guidelines](#5-new-repo-guidelines)
6. [Unified Framework Architecture](#6-unified-framework)
7. [LLM Compatibility Guidelines](#7-llm-compatibility)
8. [Computation Time Optimization](#8-performance)
9. [Summary of Recommendations](#9-recommendations)
10. [References](#10-references)
11. [Appendix A: NeuralGraph Examples](#a-neuralgraph)
12. [Appendix B: ParticleGraph Examples](#b-particlegraph)
13. [Appendix C: MPM_pytorch Examples](#c-mpm)
14. [Appendix D: MetabolismGraph Examples](#d-metabolism)
15. [Appendix E: flyvis-gnn Examples](#e-flyvis)
16. [Appendix F: Complete File Inventory](#f-inventory)

---

## 1. Executive Summary

Five repositories under `/groups/saalfeld/home/allierc/Graph/` share a common GNN message-passing architecture but have diverged through copy-paste forking into domain-specific variants for neural signaling, particle physics, material deformation, metabolism, and fly visual system modeling.

### Key Metrics

| Metric | NeuralGraph | ParticleGraph | MPM_pytorch | MetabolismGraph | flyvis-gnn |
|--------|------------|---------------|-------------|-----------------|------------|
| Total lines (est.) | ~25,000 | ~30,000 | ~15,000 | ~10,000 | ~28,000 |
| God files (>500 lines) | 16 | 12+ | 6+ | 7 | 13 |
| Largest file | 8,437 | 8,601 | 5,140 | 2,254 | 9,829 |
| Dispatch chains | 5+ (70+ branches) | 7+ | 5+ | 3 | 5+ (60+ sites) |
| Config unpacking lines | ~130 | ~150 | ~100 | ~30 | ~110 |
| Test files | 0 | 0 | 0 | 0 | 0 |
| Hardcoded paths | 15+ | 10+ | 10+ | 6+ | 10+ |
| Wildcard imports | 48 | 30+ | 20+ | 10+ | 25+ |

**Core finding**: An estimated 60-70% of code across repos is shared infrastructure that has been copy-pasted. Only 30-40% is genuinely domain-specific. This report identifies 10 cross-cutting engineering issues (Section 2), proposes two master-level core components (Section 3), and provides actionable guidelines for refactoring (Section 4), new development (Section 5), unification (Section 6), and LLM compatibility (Section 7). Specific code examples with file paths and line numbers are in Appendices A-E.

---

## 2. The 10 Engineering Issues

Each issue is described here in principle. Concrete examples with file paths and line numbers are collected in Appendices A-E.

### Issue 1: Copy-Paste Forks (Severity: Critical)

The repos are copy-paste forks of each other. Identical functions exist across repos with minor variations (e.g., `linear_model()` defined 3 times in NeuralGraph alone, `to_numpy()` defined 2 times with different bug-fixing levels, `get_index_particles()` with incompatible signatures). MetabolismGraph carries 400+ lines of dead code from NeuralGraph.

**Violated principle**: DRY -- Don't Repeat Yourself [Ref 1]. Shared code should live in a single installable package.

### Issue 2: God Files (Severity: Critical)

Multiple files exceed thousands of lines, mixing training, testing, plotting, and utilities. `GNN_PlotFigure.py` reaches 9,829 lines in flyvis-gnn. A single function `plot_signal()` is 2,010 lines.

**Violated principle**: Single Responsibility Principle (SRP) [Ref 2]. Each module should have one reason to change. A file mixing training loops and plotting has at least two.

**PyTorch best practice**: PyTorch Lightning [Ref 3] recommends separating model definition, training logic, data loading, and callbacks into distinct modules.

### Issue 3: If/Elif Model Dispatch Chains (Severity: High)

Model selection relies on string matching (`if model_name == 'PDE_N5'`) creating 15-30 branch chains scattered across 5+ files. Adding a new model requires modifying every dispatch site.

**Violated principle**: Open/Closed Principle (OCP) [Ref 2]. Software entities should be open for extension but closed for modification. The registry pattern [Ref 4] solves this.

**PyTorch best practice**: `torchvision.models` uses a registry pattern. PyTorch itself uses `torch.nn.Module` subclassing to avoid dispatch chains.

### Issue 4: Config Variable Unpacking (Severity: Medium)

Every major function begins with 20-50 lines of `variable = config.section.field` boilerplate, duplicated across train/test/plot functions.

**Violated principle**: Don't Repeat Yourself [Ref 1]. The Pydantic config objects should be used directly, or a shared context dataclass should unpack once.

### Issue 5: Magic Tensor Indexing (Severity: High)

The state tensor `x` has 7-14 columns accessed by raw integer indices (`x[:, 3:4]`, `x[:, 10:13]`). The column layout is never documented as constants and varies between domains.

**Violated principle**: Named Constants over Magic Numbers [Ref 5]. PyTorch best practice: use `NamedTuple`, dataclass, or at minimum named constants for tensor dimensions [Ref 6].

### Issue 6: Hardcoded Paths (Severity: High)

50+ absolute paths to a specific user's home directory are hardcoded across the five repos. SSH commands embed usernames. Scratch directories embed user IDs.

**Violated principle**: Configuration Externalization [Ref 7]. Paths should come from environment variables, config files, or command-line arguments.

### Issue 7: Dead Code (Severity: High)

Massive commented-out blocks (1,469 lines in one file), functions that crash if called (undefined variables), `if False:` guards over 328 lines, unused imports, and temporary files pollute the codebases.

**Violated principle**: YAGNI -- You Aren't Gonna Need It [Ref 1]. Dead code should be deleted. Version control (git) preserves history.

### Issue 8: Zero Test Infrastructure (Severity: Critical)

None of the five repositories has any test infrastructure. No `tests/` directories, no `test_*.py` files, no CI/CD configuration. Zero unit tests, zero integration tests.

**Violated principle**: Test-Driven Development [Ref 8]. At minimum, every model's forward pass, every loss function, and every data loader should have a smoke test.

**PyTorch best practice**: PyTorch repositories use `pytest` with fixtures for device management [Ref 9]. PyTorch Lightning enforces testability through its structured approach.

### Issue 9: Performance Anti-Patterns (Severity: Medium-High)

Python `for` loops over neurons (up to 14,000) calling individual model forward passes instead of batching. Repeated `torch.tensor()` creation from numpy inside training loops. Gratuitous `time.sleep()` calls. Un-vectorized mesh computations.

**Violated principle**: Vectorize, Don't Loop [Ref 10]. PyTorch and numpy operations should operate on entire tensors, not element-by-element in Python loops.

### Issue 10: Plotting Embedded in Training (Severity: Medium)

Plotting logic is interleaved with training logic. `graph_trainer.py` files contain 300-700 lines of inline matplotlib code.

**Violated principle**: Separation of Concerns [Ref 2]. PyTorch Lightning uses callbacks for plotting [Ref 3]. Training code should emit metrics; plotting code should consume them.

---

## 3. Core Architecture: Two Master-Level Components

The entire framework rests on two pillars. Everything else (plotting, config, exploration) is secondary.

### 3.1 Pillar 1: GNN Class Hierarchy

#### The Fundamental Equation

All five repos implement the same abstraction:

```
dx_i/dt = f(x_i, sum_j g(x_i, x_j, e_ij))
```

where `g` = edge message MLP (`lin_edge`), `f` = node update MLP (`lin_phi`).

| Domain | Entities | g (edge message) | f (node update) |
|--------|----------|-------------------|-----------------|
| Particle | particles | force(distance, types) | Newton's law |
| Neural | neurons | synaptic_current(voltage, types) | neural ODE |
| Mesh | mesh nodes | Laplacian(displacement) | wave/diffusion |
| MPM | material points | stress(deformation) | constitutive law |
| Metabolism | metabolites | reaction_rate(concentration) | mass-action |

#### Design: Abstract Base + Domain Subclasses

Following PyTorch Geometric's `MessagePassing` contract [Ref 11] and PyTorch's `nn.Module` design philosophy [Ref 12]:

```python
# graphlearn/models/base.py
class BaseGraphModel(MessagePassing):
    """Base for ALL graph models. Subclasses implement build_edge_features() and build_update_features()."""

    def __init__(self, config, device, aggr="add"):
        super().__init__(aggr=aggr)
        self.a = nn.Parameter(...)        # Per-node embeddings
        self.lin_edge = MLP(...)           # Edge message MLP
        self.lin_phi = MLP(...)            # Node update MLP

    @abstractmethod
    def build_edge_features(self, x_i, x_j, edge_attr) -> torch.Tensor: ...

    @abstractmethod
    def build_update_features(self, x, aggr_out) -> torch.Tensor: ...

    def message(self, x_i, x_j, edge_attr=None):
        return self.lin_edge(self.build_edge_features(x_i, x_j, edge_attr))

    def update(self, aggr_out, x):
        return self.lin_phi(self.build_update_features(x, aggr_out))
```

Each domain subclass (30-80 lines) only defines its unique feature construction:

```python
# graphlearn_neural/models/flyvis.py
@register_model("flyvis_v1")
class FlyVisModel(BaseGraphModel):
    def build_edge_features(self, x_i, x_j, edge_attr):
        activity_j = NeuronState.get(x_j, "activity")
        emb_i = self.a[NeuronState.get(x_i, "node_id").long()]
        emb_j = self.a[NeuronState.get(x_j, "node_id").long()]
        return torch.cat([activity_j, emb_i, emb_j], dim=1)
```

**Model Registry** replaces all dispatch chains (one `@register_model` decorator per model, zero if/elif):

```python
# graphlearn/models/registry.py
_REGISTRY: dict[str, type] = {}

def register_model(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def create_model(name: str, config, device) -> BaseGraphModel:
    return _REGISTRY[name](config=config, device=device)
```

This follows the Factory Method pattern [Ref 13] and matches how `torchvision.models` and `timm` [Ref 14] handle model registration.

**MLP** (single implementation, shared by all domains):

```python
# graphlearn/models/mlp.py
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128,
                 n_layers=3, activation=nn.SiLU, layer_norm=False, device=None):
        super().__init__()
        layers = []
        if n_layers == 0:
            layers.append(nn.Linear(input_size, output_size))
        else:
            layers.append(nn.Linear(input_size, hidden_size))
            if layer_norm: layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation())
            for _ in range(n_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                if layer_norm: layers.append(nn.LayerNorm(hidden_size))
                layers.append(activation())
            layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)
```

### 3.2 Pillar 2: Data Management (State Tensors & DataLoaders)

#### Named State Tensors

The current `x[:, 3:4]` pattern is replaced with named column access. This follows the principle of "making illegal states unrepresentable" [Ref 15]:

```python
# graphlearn/core/state.py
class StateLayout:
    """Base class defining named tensor columns."""
    COLUMNS: ClassVar[list[tuple[str, int]]] = []  # Override in subclass

    @classmethod
    def get(cls, x: torch.Tensor, name: str) -> torch.Tensor:
        """Get named column(s). Example: NeuronState.get(x, 'activity')"""
        return x[:, cls._col_slice(name)]

    @classmethod
    def set(cls, x: torch.Tensor, name: str, value: torch.Tensor) -> None:
        """Set named column(s) in-place."""
        x[:, cls._col_slice(name)] = value

    @classmethod
    def create(cls, n_entities: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n_entities, cls.n_columns(), device=device)
```

Domain-specific layouts:

```python
# graphlearn_neural/state.py
class NeuronState(StateLayout):
    COLUMNS = [
        ("node_id", 1), ("position", 2), ("activity", 1),
        ("external_input", 1), ("grouped_type", 1),
        ("neuron_type", 1), ("calcium", 1), ("fluorescence", 1),
    ]

# graphlearn_particles/state.py
class ParticleState(StateLayout):
    COLUMNS = [
        ("node_id", 1), ("position", 2), ("velocity", 2),
        ("type", 1), ("charge", 1),
    ]

# graphlearn_mpm/state.py
class MaterialPointState(StateLayout):
    COLUMNS = [
        ("node_id", 1), ("position", 2), ("velocity", 2),
        ("type", 1), ("C", 4), ("F", 4), ("Jp", 1),
    ]

# graphlearn_metabolism/state.py
class MetaboliteState(StateLayout):
    COLUMNS = [
        ("node_id", 1), ("position", 2), ("concentration", 1),
        ("external_input", 1), ("unused", 1), ("metabolite_type", 1),
    ]
```

#### PyTorch Dataset & DataLoader

Replace raw `x_list`/`y_list` numpy arrays with proper `torch.utils.data.Dataset` [Ref 16]:

```python
# graphlearn/data/dataset.py
class GraphTimeSeriesDataset(Dataset):
    """Dataset of graph state trajectories. Pre-converts to GPU tensors."""

    def __init__(self, data_dir, dataset_name, n_runs, n_frames,
                 time_step=1, device=torch.device("cpu")):
        self.x_list = []  # List[Tensor], pre-loaded to device
        for run in range(n_runs):
            x = np.load(data_dir / dataset_name / f"x_list_{run}.npy")
            self.x_list.append(torch.tensor(x, dtype=torch.float32, device=device))

    def __len__(self): return self.n_runs * (self.n_frames - self.time_step)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        run, frame = divmod(idx, self.n_frames - self.time_step)
        return {"x": self.x_list[run][frame], "y": self.x_list[run][frame + self.time_step]}
```

#### Edge Management

Replace scattered edge construction code with a clean `EdgeManager` class:

```python
# graphlearn/data/edges.py
class EdgeManager:
    """Manages edge indices, attributes, and per-node neighbor lists."""

    def __init__(self, edge_index, edge_attr=None, device=torch.device("cpu")):
        self.edge_index = edge_index.to(device)
        self.edge_attr = edge_attr.to(device) if edge_attr is not None else None

    @classmethod
    def from_file(cls, path, device): ...

    @classmethod
    def from_radius(cls, positions, radius, device): ...

    def build_neighbor_index(self, n_nodes) -> list[torch.Tensor]:
        """Vectorized neighbor grouping (replaces Python for-loop over nodes)."""
        sorted_idx = torch.argsort(self.edge_index[1])
        counts = torch.bincount(self.edge_index[1][sorted_idx], minlength=n_nodes)
        return torch.split(self.edge_index[0][sorted_idx], counts.tolist())
```

#### Clean Training Loop

Following PyTorch Lightning's trainer design [Ref 3] and the "thin controller" pattern [Ref 17]:

```python
# graphlearn/training/trainer.py
class Trainer:
    """Generic graph model trainer. Domain-specific behavior via callbacks."""

    def __init__(self, model, dataset, edges, optimizer, loss_fn, device, callbacks=None):
        self.model = model
        self.dataset = dataset
        self.edges = edges
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks or []

    def train(self, n_epochs, batch_size=1) -> dict[str, list[float]]:
        for epoch in range(n_epochs):
            for cb in self.callbacks: cb.on_epoch_start(self, epoch)
            epoch_loss = self._train_epoch(batch_size)
            for cb in self.callbacks: cb.on_epoch_end(self, epoch, {"loss": epoch_loss})
        return self.history

    def _train_epoch(self, batch_size) -> float: ...
```

---

## 4. Refactoring Guidelines

### 4.1 Immediate Cleanup (1-2 days per repo)

1. Delete all `if False:` blocks, commented-out code (>5 lines), dead functions, unused imports, `tmp.py`
2. Remove irrelevant domain code from forks
3. Fix actual bugs (undefined function calls, `from turtle import pos`)
4. Remove gratuitous `time.sleep()` calls
5. Replace `from X import *` with explicit imports [Ref 18]

### 4.2 Structural Refactoring (1 week per repo)

1. Define tensor column constants (StateLayout subclass per domain)
2. Extract config unpacking into shared helper or use config objects directly
3. Split god files: `graph_trainer.py` -> `trainer.py` + `evaluator.py` + `data_loader.py`
4. Split `GNN_PlotFigure.py` -> `plots/signal.py` + `plots/connectivity.py` + `plots/embedding.py`
5. Move hardcoded paths to environment variables

### 4.3 Architectural Migration (Phase 1-5)

| Phase | Scope | Effort | Deliverable |
|-------|-------|--------|-------------|
| 1 | Extract `graphlearn` core | 1-2 weeks | Shared package: MLP, SIREN, config, registry, utils, paths |
| 2 | Refactor NeuralGraph/flyvis-gnn | 1-2 weeks | `graphlearn-neural` with zero dispatch chains |
| 3 | Port remaining repos | 2-3 weeks | `graphlearn-particles`, `graphlearn-mpm`, `graphlearn-metabolism` |
| 4 | Add tests + CI | 1 week | pytest suite, GitHub Actions |
| 5 | LLM compatibility layer | 1 week | Structured configs, clean interfaces |

---

## 5. New Repository Development Guidelines

When creating a new domain variant, follow these rules:

### 5.1 Project Structure (Canonical)

```
graphlearn_{domain}/
├── config/                    # YAML configs
├── src/graphlearn_{domain}/
│   ├── config.py              # Domain config (extends BaseConfig)
│   ├── state.py               # Named state columns
│   ├── models/                # @register_model classes
│   ├── generators/            # Data generation
│   ├── training/              # Domain-specific overrides
│   └── plotting/              # Domain-specific plots
├── tests/                     # pytest tests
└── pyproject.toml             # depends on graphlearn
```

### 5.2 Critical Rules

| Rule | Principle | Reference |
|------|-----------|-----------|
| Registry, not dispatch chains | Open/Closed [Ref 2] | `torchvision.models` |
| Named state columns, not magic indices | Named Constants [Ref 5] | `NamedTuple` pattern |
| Config objects, not unpacking | DRY [Ref 1] | Pydantic [Ref 19] |
| Separate train / evaluate / plot | SRP [Ref 2] | PyTorch Lightning [Ref 3] |
| Tests from day one | TDD [Ref 8] | `pytest` [Ref 9] |
| No hardcoded paths | Config Externalization [Ref 7] | 12-Factor App [Ref 20] |
| Files < 500 lines | Readability [Ref 21] | Google Python Style |
| Functions < 100 lines | Readability [Ref 21] | Google Python Style |
| Explicit imports only | PEP 8 [Ref 18] | `from x import y` |
| Type annotations | PEP 484 [Ref 22] | `mypy` |

---

## 6. Unified Framework Architecture

### 6.1 Why Unify?

All five repos model: **entities on a graph exchanging messages and evolving in time**. Only the physics/biology differ. The shared infrastructure (60-70% of code) should be a single package.

### 6.2 Package Structure

```
graphlearn/                          # Shared core (pip-installable)
├── core/                            # Config, registry, state, paths, device
├── models/                          # MLP, SIREN, BaseGraphModel, factory
├── training/                        # Trainer, Evaluator, Loss, Callbacks
├── exploration/                     # UCB tree, LLM interface
├── plotting/                        # Plot engine base, loss/embedding/function plots
├── data/                            # Dataset, DataLoader, EdgeManager
└── utils.py                         # to_numpy, linear_model (ONE copy)

graphlearn-neural/                   # NeuralGraph + flyvis-gnn replacement
graphlearn-particles/                # ParticleGraph replacement
graphlearn-mpm/                      # MPM_pytorch replacement
graphlearn-metabolism/               # MetabolismGraph replacement
```

### 6.3 Key Design Patterns

**Composable Loss** (replaces inline loss computation in training loops):

```python
class CompositeLoss:
    def __init__(self, terms: list[LossTerm]):
        self.terms = terms
    def compute(self, model, pred, target) -> dict[str, torch.Tensor]:
        return {type(t).__name__: t.compute(model, pred, target) for t in self.terms}
```

**Callback-Based Events** (separates plotting from training) [Ref 3]:

```python
class Callback:
    def on_epoch_start(self, trainer, epoch): pass
    def on_batch_end(self, trainer, batch, loss): pass
    def on_epoch_end(self, trainer, epoch, metrics): pass

class PlotCallback(Callback): ...
class CheckpointCallback(Callback): ...
class LogCallback(Callback): ...
```

**Composable Physics** (replaces 48 PDE_D_*.py files):

```python
class PhysicsTerm:
    def compute(self, x_i, x_j, edge_attr) -> torch.Tensor: ...

class Damping(PhysicsTerm): ...
class CIL(PhysicsTerm): ...
class Durotaxis(PhysicsTerm): ...

@register_model("cell_custom")
class CompositeCellModel(BaseGraphModel):
    def __init__(self, terms: list[PhysicsTerm], **kwargs):
        self.terms = terms
    def message(self, x_i, x_j, edge_attr):
        return sum(t.compute(x_i, x_j, edge_attr) for t in self.terms)
```

---

## 7. LLM Compatibility Guidelines

Making code LLM-friendly means making it **readable, modular, and predictable** [Ref 23].

### 7.1 Size Constraints

| Constraint | Limit | Rationale |
|-----------|-------|-----------|
| File length | < 500 lines | LLM context window efficiency |
| Function length | < 100 lines | LLM comprehension |
| Class length | < 300 lines | Single responsibility |

### 7.2 Explicit Over Implicit

Replace magic numbers, wildcard imports, and implicit conventions with named, greppable identifiers. This applies the Principle of Least Surprise [Ref 24].

### 7.3 Self-Documenting Config

Config files should be the single source of truth. An LLM should read the config and understand the entire experiment [Ref 19].

### 7.4 Registry-Based Extension

Adding new functionality = ONE new file + ONE `@register_model` decorator. Zero changes to existing files. This is the most LLM-compatible pattern: the LLM only needs to read the base class and write a new subclass.

### 7.5 Predictable Structure

Every domain package follows identical directory structure. An LLM that has worked with one domain can immediately navigate any other.

### 7.6 Type Annotations

Type hints enable LLM comprehension and tool support (`mypy`, IDEs) [Ref 22].

### 7.7 LLM-Modifiable Interface

```python
class LLMModifiable:
    def get_modifiable_parameters(self) -> dict[str, tuple[float, float]]: ...
    def describe(self) -> str: ...
```

### 7.8 Tests as LLM Guardrails

Tests give the LLM confidence that modifications are correct. After every LLM-proposed change: `pytest tests/ -x --timeout=60`.

---

## 8. Computation Time Optimization

### 8.1 Profiling (Currently Missing)

Add `torch.profiler` integration and simple timing context managers.

### 8.2 Vectorization Checklist

| Anti-pattern | Fix |
|-------------|-----|
| `for n in range(n_neurons): model.lin_edge(...)` | Batch: `model.lin_edge(all_features)` |
| `torch.tensor(numpy_array)` in training loop | Pre-convert once before loop |
| `for i in range(n): mask = edges[1] == i` | `torch.scatter` / `torch.bincount` |
| `x.detach().cpu().numpy()` in inner loop | Accumulate on GPU, transfer once |
| `time.sleep(...)` | Remove |

### 8.3 GPU Memory

Add `torch.cuda.memory_allocated()` logging at key points. Use `torch.cuda.amp` for mixed-precision training where applicable [Ref 25].

---

## 9. Summary of Recommendations

### Priority 1: Immediate (days)
- Delete dead code (~5,000+ lines across repos)
- Fix crashes (undefined functions, `from turtle import pos`)
- Remove `time.sleep()`, irrelevant domain code

### Priority 2: Structural (weeks)
- Define state column constants per domain
- Split god files (< 500 lines each)
- Explicit imports, move hardcoded paths to config

### Priority 3: Architecture (months)
- Create `graphlearn` shared core
- Model registry, composable loss, callback training
- Port all repos to shared core
- Add pytest test suite

### Priority 4: Quality (ongoing)
- Profiling utilities, vectorize loops
- CI/CD pipeline
- LLM compatibility interfaces
- Type annotations

---

## 10. References

### Software Engineering Principles

[Ref 1] Hunt, A. & Thomas, D. (1999). *The Pragmatic Programmer*. Addison-Wesley. DRY (Don't Repeat Yourself), YAGNI (You Aren't Gonna Need It).

[Ref 2] Martin, R.C. (2003). *Agile Software Development: Principles, Patterns, and Practices*. Prentice Hall. SOLID principles: Single Responsibility (SRP), Open/Closed (OCP), Liskov Substitution, Interface Segregation, Dependency Inversion.

[Ref 5] McConnell, S. (2004). *Code Complete*, 2nd ed. Microsoft Press. Chapter 12: Named constants over magic numbers.

[Ref 7] Humble, J. & Farley, D. (2010). *Continuous Delivery*. Addison-Wesley. Configuration externalization.

[Ref 8] Beck, K. (2002). *Test-Driven Development: By Example*. Addison-Wesley.

[Ref 13] Gamma, E. et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley. Factory Method, Registry patterns.

[Ref 17] Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley. Thin controller pattern.

[Ref 20] Wiggins, A. (2011). *The Twelve-Factor App*. https://12factor.net/. Config via environment variables.

[Ref 21] Google (2024). *Google Python Style Guide*. https://google.github.io/styleguide/pyguide.html. File/function length limits, import conventions.

[Ref 24] Raymond, E.S. (2003). *The Art of Unix Programming*. Addison-Wesley. Principle of Least Surprise.

### Python & PyTorch Best Practices

[Ref 3] Falcon, W. et al. (2019-2026). *PyTorch Lightning*. https://lightning.ai/. Trainer pattern, callbacks, separation of model/training/data.

[Ref 4] Python Registry Pattern: https://realpython.com/factory-method-python/. Factory and registry implementations in Python.

[Ref 6] PyTorch Best Practices: https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html. Module design, tensor conventions.

[Ref 9] pytest documentation: https://docs.pytest.org/. Fixtures, parametrize, device management for GPU tests.

[Ref 10] NumPy documentation: *Vectorization*. https://numpy.org/doc/stable/glossary.html. "Vectorize, don't loop."

[Ref 11] Fey, M. & Lenssen, J.E. (2019). *Fast Graph Representation Learning with PyTorch Geometric*. ICLR Workshop. MessagePassing base class.

[Ref 12] PyTorch `nn.Module` Design: https://pytorch.org/docs/stable/notes/modules.html. Subclassing conventions, parameter registration.

[Ref 14] Wightman, R. (2019-2026). *timm: PyTorch Image Models*. https://github.com/huggingface/pytorch-image-models. Registry pattern for model creation.

[Ref 15] Yaron Minsky (2011). *Making Illegal States Unrepresentable*. Blog post on type-safe data design.

[Ref 16] PyTorch Dataset/DataLoader: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html.

[Ref 18] PEP 8 -- Style Guide for Python Code: https://peps.python.org/pep-0008/. Import conventions, wildcard import prohibition.

[Ref 19] Pydantic documentation: https://docs.pydantic.dev/. Config validation, default values, type coercion.

[Ref 22] PEP 484 -- Type Hints: https://peps.python.org/pep-0484/. Function annotations, `mypy` compatibility.

[Ref 23] Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code*. arXiv:2107.03374. Code readability impact on LLM performance.

[Ref 25] PyTorch AMP (Automatic Mixed Precision): https://pytorch.org/docs/stable/amp.html.

---

## Appendix A: NeuralGraph Specific Examples

### A.1 If/Elif Dispatch Chains

**`choose_training_model()` in `src/NeuralGraph/models/utils.py:748-838`**

Triple match/case dispatch with ~30 branches:

```python
# 16-case particle dispatch (lines 758-809)
match model_name:
    case 'PDE_R':
        model = Interaction_Mouse(...)
    case 'PDE_MPM' | 'PDE_MPM_A':
        model = Interaction_MPM(...)
    case 'PDE_Cell' | 'PDE_Cell_area':
        model = Interaction_Cell(...)
    # ... 13 more cases ...

# 4-case mesh dispatch (lines 811-823)
match model_name:
    case 'DiffMesh': ...
    case 'WaveMesh': ...

# 2-case signal dispatch (lines 826-833)
match model_name:
    case 'PDE_N2' | 'PDE_N3' | ... | 'PDE_N11':
        model = Signal_Propagation(...)
```

**`get_in_features()` in `models/utils.py:149-198`**: 15-branch match/case.

**`get_in_features_lin_edge()` in `models/utils.py:108-147`**: 7-branch if/elif.

**`choose_model()` in `generators/utils.py:19-58`**: 7-branch match/case.

**`data_train()` in `graph_trainer.py:67-77`**: 5-branch if/elif on `config.dataset`.

### A.2 Config Unpacking

**`data_train_signal()` (`graph_trainer.py:80-116`)**: 37 lines of pure unpacking.
**`data_train_flyvis()` (`graph_trainer.py:767-813`)**: 47 lines.
**`data_test_signal()` (`graph_trainer.py:2191-2244`)**: ~50 lines.

### A.3 Dead Code

**`init_synapse_map()` (`generators/utils.py:304-342`)**: References undefined variables (`edge_index`, `N1`, `V1`, `T1`, `H1`, `A1`, `dataset_name`). Would crash if called.

**`generators/tmp.py`**: 874 lines of scratch code.

**`models/utils.py:31`**: Defines `linear_model()` but also imports it from `fitting_models` at line 23 (shadows import).

**Duplicate imports**: `import os` on lines 5 and 34 of `GNN_Main.py`. `import torch.nn.functional as F` on lines 8 and 30 of `graph_trainer.py`.

**48 wildcard `from X import *` statements** across the codebase.

### A.4 Magic Tensor Indexing

```python
# graph_trainer.py:144
type_list = torch.tensor(x[:, 1 + 2 * dimension:2 + 2 * dimension], device=device)

# data_loaders.py:647-652
x[:, 6] = 6          # What is column 6? Why the value 6?
x[:, 10:13] = odor_worms[run,:,it]  # Columns 10-13 = odor channels (undocumented)

# models/utils.py:1035-1043
x_baseline[:, 10:13] = 0  # no excitation
x_odor[:, 10 + i] = 1     # activate specific odor
```

### A.5 Hardcoded Paths

```python
# data_loaders.py:423
'/groups/saalfeld/home/allierc/signaling/Celegans/Cook_2019/SI_5_corrected_July_2020_bis.xlsx'

# models/utils.py:1113 (default argument)
out_prefix="/groups/saalfeld/home/allierc/Py/NeuralGraph/graphs_data/fly/"

# ODEs/Fitzhug_Nagumo.py:223
config_root = "/groups/saalfeld/home/allierc/Py/NeuralGraph/src/NeuralGraph/ODEs/config/"
```

15+ hardcoded paths in at least 6 files.

### A.6 Performance Issues

**`models/utils.py:251-268`**: Python loop over neurons calling `model.lin_edge()` one neuron at a time.

**`graph_trainer.py:424-434`**: Python loop for W_sign regularization inside training loop.

**`generators/utils.py:264-275`**: Pure Python loop over mesh faces for distance computation.

### A.7 God Files

8,437 lines (`GNN_PlotFigure.py`), 4,366 lines (`graph_trainer.py`), 2,186 lines (`utils.py`), 1,547 lines (`models/utils.py`), 1,525 lines (`graph_data_generator.py`), + 11 more files over 500 lines.

---

## Appendix B: ParticleGraph Specific Examples

### B.1 PDE Variant Explosion

**48 separate PDE_D_*.py files** for cell-tracking variants. Examples: PDE_D_DampedCTC.py, PDE_D_DualDampedCTC.py, PDE_D_InertialDampedCTC.py, PDE_D_CrossRepulsionDampedCTC.py, PDE_D_VicsekDampedCTC.py, PDE_D_AnisotropicDampedCTC.py, PDE_D_MemoryDampedCTC.py, PDE_D_FateSwitchCTC.py, PDE_D_AdaptiveRangeCTC.py, PDE_D_SaturatingCIL.py, PDE_D_GradientDampedCTC.py, PDE_D_RatioCTC.py, PDE_D_AsymmetricCTC.py, PDE_D_DensityDragCIL.py, PDE_D_DeadzoneCTC.py, PDE_D_AdaptivePF.py, PDE_D_DualFieldCTC.py, etc.

Most share ~90% identical code with 1-2 unique terms.

### B.2 God Files

5,453 lines (`graph_trainer.py`), 8,601 lines (`GNN_PlotFigure.py`).

### B.3 Config Unpacking

`data_train_particle()`: ~40 lines of unpacking. `data_train_signal()`: ~35 lines. Pattern repeated in test functions.

---

## Appendix C: MPM_pytorch Specific Examples

### C.1 Dual Trainers

**Two parallel trainers with duplicated logic**: `graph_trainer.py` (1,385 lines) and `particle_graph_trainer.py` (5,140 lines).

### C.2 Bizarre Import

```python
# Interaction_MPM.py:1
from turtle import pos   # imports from Python's turtle graphics module
```

### C.3 Duplicate Data Import

```python
# Interaction_MPM.py:9-10
import torch_geometric.data as data
import torch_geometric.data as data    # exact duplicate
```

### C.4 Inherited Dead Dispatch

`choose_training_model()` (`models/utils.py:1719-1812`): 94-line dispatch including models (Interaction_Mouse, Interaction_Cell, Signal_Propagation) irrelevant to MPM.

### C.5 Magic Tensor Indexing

```python
# utils.py:1720
F_gt = x[:, 9:13].reshape(-1, 2, 2)    # deformation gradient
# utils.py:1775
C_gt = x[:, 5:9].reshape(-1, 2, 2)     # affine velocity
# utils.py:1820
Jp_gt = x[:, 13:14]                     # determinant
```

---

## Appendix D: MetabolismGraph Specific Examples

### D.1 Dead NeuralGraph Code

**`LossRegularizer` class (`models/utils.py:404-798`)**: 395 lines referencing `model.W`, `model.WL`, `model.WR`, `model.b`, `model.lin_edge`, `model.lin_phi`, Dale's Law enforcement -- none of which exist in Metabolism_Propagation. Imported but never instantiated.

**Dead dispatch chain (`models/utils.py:856-896`)**: 40 lines of dispatch for PDE_N4/N5/N7/N8/N9/N11, all unreachable.

### D.2 Dead `if False:` Block

```python
# graph_trainer.py:697
if False: # homeostasis_training:
    # 328 lines of unreachable Phase 2 code
```

### D.3 Legacy NeuralGraph Loss Colors

```python
# generators/utils.py:107-115
('W_L1', 'r', 1.5, 'W L1 sparsity'),     # Never used in metabolism
('W_sign', 'navy', 1.5, 'W sign (Dale)'), # Dale's Law is a neural concept
('edge_diff', 'magenta', 1.5, 'MLP1 monotonicity'),
```

### D.4 Unused Functions

`set_trainable_parameters()`, `constant_batch_size()`, `increasing_batch_size()`, `fig_init()` -- all defined but never called.

### D.5 Unused Imports (graph_trainer.py)

`shutil`, `matplotlib as mpl`, `random`, `torch_geometric as pyg`, `DataLoader`, `LossRegularizer`, `fig_init` -- 7 unused imports.

---

## Appendix E: flyvis-gnn Specific Examples

### E.1 Dispatch Chains (60+ sites)

**Model construction** (`graph_trainer.py:287-294, 1026-1031, 1794-1803`): 4-6 branch chains for Signal_Propagation variants.

**GNN_PlotFigure.py**: 60+ sites matching `PDE_N4/N5/N7/N8/N11` for feature construction.

**`get_log_dir()` (`utils.py:518-548`)**: 12-way chain for PDE_A/B/E/F/G/K/GS/RD/Wave -- none relevant to fly visual system.

### E.2 Undefined Functions

**`plot_synaptic_flyvis_calcium()`** (`GNN_PlotFigure.py:8826`): Called but never defined. `# noqa: F821` acknowledges the error.

**`data_train_zebra()`** (`graph_trainer.py:168`): Called but never defined in this repo.

### E.3 Inherited Irrelevant Code (~3,500+ lines)

- `plot_synaptic_CElegans()`: 1,837 lines for C. elegans (not fly)
- `plot_synaptic_zebra()`: 384 lines for zebrafish
- Particle/mesh dispatch chains in `utils.py` and `models/utils.py`: ~500 lines
- Config fields: `particle_model_name`, `cell_model_name`, `mesh_model_name`, `min_radius`, `max_radius`, `boundary`

### E.4 Commented-Out Code

1,469 lines of commented-out code in `GNN_PlotFigure.py` (15% of the 9,829-line file). Largest block: lines 4826-5596 (~770 continuous lines).

### E.5 Performance Issues

**`graph_trainer.py:366-369`**: Python loop over ~14,000 neurons for edge index construction.

**`graph_trainer.py:433`**: `torch.tensor(x_list[run][k])` called every training iteration instead of pre-converting.

**`graph_trainer.py:275, 392`**: `time.sleep(0.5)` and `time.sleep(0.2)` waste 0.7s per training run.

### E.6 God Files

9,829 lines (`GNN_PlotFigure.py`), 2,904 lines (`models/utils.py`), 2,675 lines (`graph_trainer.py`), 2,234 lines (`utils.py`), + 9 more files over 500 lines.

---

## Appendix F: Complete God File Inventory

### All files over 500 lines across the five repos

| Lines | Repo | File |
|------:|------|------|
| 9,829 | flyvis-gnn | `GNN_PlotFigure.py` |
| 8,601 | ParticleGraph | `GNN_PlotFigure.py` |
| 8,437 | NeuralGraph | `GNN_PlotFigure.py` |
| 5,453 | ParticleGraph | `models/graph_trainer.py` |
| 5,140 | MPM_pytorch | `models/particle_graph_trainer.py` |
| 4,366 | NeuralGraph | `models/graph_trainer.py` |
| 2,904 | flyvis-gnn | `models/utils.py` |
| 2,675 | flyvis-gnn | `models/graph_trainer.py` |
| 2,254 | MetabolismGraph | `models/graph_trainer.py` |
| 2,234 | flyvis-gnn | `utils.py` |
| 2,186 | NeuralGraph | `utils.py` |
| 1,680 | flyvis-gnn | `generators/graph_data_generator.py` |
| 1,547 | NeuralGraph | `models/utils.py` |
| 1,525 | NeuralGraph | `generators/graph_data_generator.py` |
| 1,385 | MPM_pytorch | `models/graph_trainer.py` |
| 1,258 | MetabolismGraph | `GNN_LLM_phase2.py` |
| 1,237 | NeuralGraph | `models/plot_utils.py` |
| 1,217 | flyvis-gnn | `generators/utils.py` |
| 1,215 | flyvis-gnn | `models/plot_utils.py` |
| 1,154 | flyvis-gnn | `models/exploration_tree.py` |
| 1,125 | MetabolismGraph | `models/exploration_tree.py` |
| 1,094 | MetabolismGraph | `models/utils.py` |
| 1,080 | flyvis-gnn | `generators/davis.py` |
| 1,000 | NeuralGraph | `generators/davis.py` |
| 995 | NeuralGraph | `data_loaders.py` |
| 970 | flyvis-gnn | `GNN_LLM_parallel.py` |
| 954 | NeuralGraph | `models/Ising_analysis.py` |
| 949 | MetabolismGraph | `GNN_LLM_parallel.py` |
| 933 | flyvis-gnn | `GNN_LLM_parallel_flyvis.py` |
| 874 | NeuralGraph | `generators/tmp.py` |
| 860 | flyvis-gnn | `GNN_LLM.py` |
| 806 | MetabolismGraph | `GNN_LLM.py` |

---

*Report generated by Claude (Opus 4.6) on 2026-02-14. Based on direct code analysis of 5 repositories totaling ~108,000 lines.*

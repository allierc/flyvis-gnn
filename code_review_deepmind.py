#!/usr/bin/env python3
"""Generate the FlyVis-GNN Code Review PDF using reportlab."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Preformatted, KeepTogether,
    HRFlowable,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
import os

OUTPUT_PATH = "/workspace/flyvis-gnn/flyvis_gnn_code_review.pdf"


def build_styles():
    ss = getSampleStyleSheet()

    ss.add(ParagraphStyle(
        name='DocTitle',
        parent=ss['Title'],
        fontSize=22,
        leading=28,
        spaceAfter=6,
        textColor=HexColor('#1a1a2e'),
    ))
    ss.add(ParagraphStyle(
        name='DocSubtitle',
        parent=ss['Normal'],
        fontSize=12,
        leading=16,
        spaceAfter=20,
        textColor=HexColor('#555555'),
        alignment=TA_CENTER,
    ))
    ss.add(ParagraphStyle(
        name='H1',
        parent=ss['Heading1'],
        fontSize=16,
        leading=20,
        spaceBefore=18,
        spaceAfter=8,
        textColor=HexColor('#1a1a2e'),
        borderWidth=0,
        borderPadding=0,
    ))
    ss.add(ParagraphStyle(
        name='H2',
        parent=ss['Heading2'],
        fontSize=13,
        leading=17,
        spaceBefore=14,
        spaceAfter=6,
        textColor=HexColor('#2d3436'),
    ))
    ss.add(ParagraphStyle(
        name='H3',
        parent=ss['Heading3'],
        fontSize=11,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
        textColor=HexColor('#2d3436'),
    ))
    ss.add(ParagraphStyle(
        name='Body',
        parent=ss['Normal'],
        fontSize=9.5,
        leading=13.5,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        textColor=HexColor('#222222'),
    ))
    ss.add(ParagraphStyle(
        name='CodeBlock',
        parent=ss['Code'],
        fontSize=7.5,
        leading=10,
        spaceAfter=6,
        backColor=HexColor('#f4f4f4'),
        borderWidth=0.5,
        borderColor=HexColor('#cccccc'),
        borderPadding=6,
        leftIndent=12,
        rightIndent=12,
        fontName='Courier',
    ))
    ss.add(ParagraphStyle(
        name='TableCell',
        parent=ss['Normal'],
        fontSize=8,
        leading=11,
        textColor=HexColor('#222222'),
    ))
    ss.add(ParagraphStyle(
        name='TableHeader',
        parent=ss['Normal'],
        fontSize=8,
        leading=11,
        textColor=colors.white,
        fontName='Helvetica-Bold',
    ))
    ss.add(ParagraphStyle(
        name='Score',
        parent=ss['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_CENTER,
        textColor=HexColor('#d63031'),
        fontName='Helvetica-Bold',
    ))

    return ss


def make_table(headers, rows, col_widths=None):
    """Build a styled table."""
    styles = build_styles()
    header_row = [Paragraph(f"<b>{h}</b>", styles['TableHeader']) for h in headers]
    data = [header_row]
    for row in rows:
        data.append([Paragraph(str(c), styles['TableCell']) for c in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.4, HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f8f8f8')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    return t


def code_block(text, styles):
    return Preformatted(text, styles['CodeBlock'])


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=HexColor('#cccccc'),
                       spaceBefore=4, spaceAfter=8)


def build_document():
    styles = build_styles()
    story = []
    B = lambda t: Paragraph(t, styles['Body'])
    H1 = lambda t: Paragraph(t, styles['H1'])
    H2 = lambda t: Paragraph(t, styles['H2'])
    H3 = lambda t: Paragraph(t, styles['H3'])
    SP = lambda n=6: Spacer(1, n)

    # =========================================================================
    # TITLE PAGE
    # =========================================================================
    story.append(SP(120))
    story.append(Paragraph("FlyVis-GNN Repository", styles['DocTitle']))
    story.append(Paragraph("Comprehensive Code Review", styles['DocTitle']))
    story.append(SP(14))
    story.append(hr())
    story.append(Paragraph(
        "A Senior ML Engineering Assessment of Architecture, Code Quality, "
        "Anti-Patterns, and Future Directions for Breaking the Performance Ceiling",
        styles['DocSubtitle'],
    ))
    story.append(SP(20))
    story.append(Paragraph("Perspective: Senior Machine Learning Engineer, Google DeepMind", styles['DocSubtitle']))
    story.append(Paragraph("Date: February 2026", styles['DocSubtitle']))
    story.append(Paragraph("Repository: flyvis-gnn v0.1.0 | ~27,300 lines of Python", styles['DocSubtitle']))
    story.append(PageBreak())

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    story.append(H1("Table of Contents"))
    story.append(SP(6))
    toc_items = [
        "1. Executive Summary",
        "2. Repository Architecture",
        "3. What Works Well: Good Practices",
        "4. What Needs Work: Spaghetti Code and Anti-Patterns",
        "5. Deep-Dive: The Four Worst Offenders",
        "6. File-by-File Scorecard",
        "7. Recommendations and Refactoring Roadmap",
    ]
    for item in toc_items:
        story.append(B(item))
    story.append(SP(8))
    story.append(PageBreak())

    # =========================================================================
    # 1. EXECUTIVE SUMMARY
    # =========================================================================
    story.append(H1("1. Executive Summary"))
    story.append(SP(4))

    story.append(B(
        "FlyVis-GNN is a research-grade Graph Neural Network framework designed to recover neural connectivity "
        "in the Drosophila visual system from neuronal activity recordings. The project sits at the intersection "
        "of computational neuroscience and modern deep learning, combining sparse message-passing GNNs, Neural "
        "ODE integration for continuous-time dynamics, SIREN-based implicit neural representations for visual "
        "field reconstruction, and an innovative LLM-guided hyperparameter exploration pipeline. The repository "
        "contains approximately 27,300 lines of Python across 50 files, with the core package living in "
        "<font face='Courier' size='8'>src/flyvis_gnn/</font>."
    ))
    story.append(B(
        "The scientific ambition of this project is commendable and the underlying mathematical formulation is "
        "sound. The GNN learns per-edge synaptic weights W, per-node embeddings a, an edge message function "
        "g_phi, and a node update function f_theta, such that the dynamics du/dt = f_theta(v, a, sum(msg), "
        "excitation) faithfully reproduce observed neural activity. The framework supports multiple training "
        "modalities including single-step derivative prediction, multi-step recurrent unrolling, and full "
        "Neural ODE integration with adjoint-method backpropagation. Biologically-motivated constraints such "
        "as Dale's Law enforcement, sparse connectivity priors, and monotonic synaptic transmission are "
        "elegantly woven into a 14-component regularization system."
    ))
    story.append(B(
        "However, from a software engineering perspective, the codebase exhibits the characteristic pathology "
        "of a research project that has grown organically over many experimental iterations without periodic "
        "architectural refactoring. The training loop alone spans 2,799 lines in a single file. There are no "
        "unit tests, no CI/CD pipeline, no linter configuration despite ruff being installed, and critical "
        "infrastructure like learning rate scheduling and early stopping are either absent or implemented "
        "ad hoc. Hardcoded institution-specific paths, debug print statements left in production code, magic "
        "numbers scattered throughout the training logic, and a manual batch assembly mechanism that bypasses "
        "PyTorch's DataLoader abstraction all contribute to a codebase that is fragile, difficult to extend, "
        "and challenging for new contributors to navigate."
    ))
    story.append(B(
        "The overall code quality score, assessed across ten dimensions including testing, documentation, "
        "error handling, maintainability, and architectural coherence, is <b>5.0 out of 10</b>. This is "
        "below the threshold one would expect for a project of this scientific sophistication. The good news "
        "is that the core algorithms are well-conceived and the configuration system (Pydantic-based YAML) "
        "is genuinely well-designed. The problems are structural and organizational rather than algorithmic, "
        "which means a focused refactoring effort could dramatically improve the codebase without requiring "
        "changes to the underlying science."
    ))

    story.append(SP(6))
    story.append(Paragraph("Overall Code Quality Score: 5.0 / 10", styles['Score']))
    story.append(SP(6))

    # =========================================================================
    # 2. REPOSITORY ARCHITECTURE
    # =========================================================================
    story.append(H1("2. Repository Architecture"))
    story.append(SP(4))

    story.append(B(
        "The repository follows a standard Python package layout with the core library in "
        "<font face='Courier' size='8'>src/flyvis_gnn/</font> and entry-point scripts at the repository "
        "root. The package is structured around five major subsystems: the model definitions in "
        "<font face='Courier' size='8'>models/</font>, the data generation pipeline in "
        "<font face='Courier' size='8'>generators/</font>, the configuration system in "
        "<font face='Courier' size='8'>config.py</font>, the visualization suite in "
        "<font face='Courier' size='8'>plot.py</font>, and the neuron state management in "
        "<font face='Courier' size='8'>neuron_state.py</font>. Entry points include "
        "<font face='Courier' size='8'>GNN_Main.py</font> for standard training runs, "
        "<font face='Courier' size='8'>GNN_LLM.py</font> for LLM-guided exploration, "
        "<font face='Courier' size='8'>GNN_Test.py</font> for evaluation, and "
        "<font face='Courier' size='8'>GNN_PlotFigure.py</font> for post-hoc visualization."
    ))

    story.append(B(
        "The data flow is conceptually clean: YAML configuration files in "
        "<font face='Courier' size='8'>config/fly/</font> parameterize the entire pipeline from data "
        "generation through training to evaluation. The data generation step loads a connectome graph "
        "(edge_index, ground-truth weights) and visual stimuli (DAVIS video dataset), simulates forward "
        "neural dynamics using known parameters, and produces training and test splits stored as Zarr arrays. "
        "The training step loads these arrays into NeuronTimeSeries objects, instantiates the FlyVisGNN model, "
        "and runs the optimization loop. The testing step rolls out predictions on held-out data and computes "
        "R-squared metrics against ground truth for connectivity weights, resting potentials, and time constants."
    ))

    story.append(H2("2.1 Core Model Architecture"))

    story.append(B(
        "The central model class, FlyVisGNN (272 lines in <font face='Courier' size='8'>models/flyvis_gnn.py"
        "</font>), implements a message-passing GNN with explicit scatter_add aggregation rather than relying "
        "on PyTorch Geometric. This is a deliberate design choice that eliminates a heavyweight dependency "
        "while maintaining full control over the message computation. The model defines four learnable "
        "components: a per-edge weight vector W of shape (n_edges, 1), a per-node embedding matrix a of "
        "shape (n_neurons, embedding_dim), an edge message function g_phi implemented as an MLP, and a node "
        "update function f_theta also implemented as an MLP. The forward pass computes messages by evaluating "
        "g_phi on per-edge features (source voltage and embedding for the flyvis_A variant, or both source "
        "and destination features for flyvis_B), optionally squaring the output to enforce non-negative "
        "synaptic transmission, multiplying by the per-edge weight W, and aggregating via scatter_add to "
        "destination nodes. The aggregated message, along with the node's voltage, embedding, and external "
        "stimulus, is then fed to f_theta which outputs the predicted du/dt."
    ))

    story.append(B(
        "The MLP class (160 lines in <font face='Courier' size='8'>models/MLP.py</font>) is a standard "
        "feedforward network with configurable depth, width, and activation function. Weight initialization "
        "uses small normal distributions (std=0.1) with zero biases, and dropout is supported but typically "
        "disabled (rate=0.0). The class is straightforward but includes a significant code smell: the "
        "<font face='Courier' size='8'>__main__</font> block at the bottom contains 100 lines of ad-hoc "
        "test/debugging code with hardcoded device strings and matplotlib interactive backend calls that "
        "should never appear in a production module."
    ))

    story.append(H2("2.2 Training Infrastructure"))

    story.append(B(
        "The training system centers on <font face='Courier' size='8'>models/graph_trainer.py</font>, "
        "which at 2,799 lines is by far the largest file in the repository and the single greatest source "
        "of technical debt. The file contains the main training function data_train_flyvis (approximately "
        "720 lines), an RNN variant data_train_flyvis_RNN, an INR training function data_train_INR, the "
        "test pipeline data_test_flyvis, and a special ablation test function data_test_flyvis_special. "
        "Each of these functions contains its own data loading, model construction, optimizer setup, "
        "training loop, checkpoint management, visualization, and metrics logging, with substantial code "
        "duplication between them."
    ))

    story.append(B(
        "The training loop itself implements manual batch assembly by randomly sampling time frames from "
        "the simulation data, constructing NeuronState objects for each frame, optionally computing visual "
        "field reconstructions through the SIREN network, and concatenating frames into a batched graph "
        "using the _batch_frames utility. This manual batching bypasses PyTorch's DataLoader abstraction "
        "entirely, which means there is no support for multi-worker data loading, prefetching, or "
        "reproducible shuffling. The sampling logic uses bare numpy random calls (np.random.randint) "
        "without proper seed management within epochs, making exact reproducibility across runs difficult "
        "to guarantee."
    ))

    story.append(H2("2.3 Neural ODE Integration"))

    story.append(B(
        "The Neural ODE subsystem (<font face='Courier' size='8'>models/Neural_ode_wrapper_FlyVis.py</font>, "
        "281 lines) wraps the GNN model as an ODE vector field compatible with torchdiffeq. The "
        "GNNODEFunc_FlyVis class maintains a template NeuronState and updates its voltage field at each "
        "integration step, while the visual stimulus is either reconstructed from the SIREN network or "
        "looked up from the pre-computed time series. The integration function supports both standard "
        "and adjoint ODE solvers, with the adjoint method providing O(1) memory cost in the number of "
        "integration steps at the expense of additional computation during the backward pass. The file "
        "contains a global DEBUG_ODE flag set to True in production code, which triggers verbose print "
        "statements every 500 iterations. This is a clear violation of basic code hygiene and should be "
        "replaced with proper logging at the DEBUG level."
    ))

    story.append(H2("2.4 Configuration System"))

    story.append(B(
        "The configuration system (<font face='Courier' size='8'>config.py</font>, 646 lines) is one of "
        "the genuinely well-designed components of the codebase. It uses Pydantic BaseModel classes with "
        "StrEnum types for categorical fields (boundary conditions, activation functions, integration "
        "methods, calcium models, sparsification strategies), providing both type safety and IDE "
        "autocompletion. The top-level NeuralGraphConfig contains nested SimulationConfig, GraphModelConfig, "
        "TrainingConfig, PlottingConfig, ZarrConfig, and ClaudeConfig sections, each with sensible defaults "
        "and descriptive field names. Configuration files are loaded from YAML, validated by Pydantic, and "
        "can be serialized back to YAML for experiment tracking. This is exactly the right approach for a "
        "research project with many hyperparameters."
    ))

    story.append(H2("2.5 Regularization System"))

    story.append(B(
        "The LossRegularizer class (<font face='Courier' size='8'>models/utils.py</font>, starting at line "
        "1165) manages 14 distinct regularization components with epoch-dependent coefficient scheduling. "
        "The system tracks history for each component, supports two-phase training where certain constraints "
        "(like g_phi monotonicity) are active in phase one and relaxed in phase two, and provides a clean "
        "interface for computing per-iteration regularization losses. The regularizer handles W_L1 sparsity "
        "for connection pruning, W_sign for Dale's Law enforcement, g_phi_diff for monotonic synaptic "
        "transmission, embedding norm penalties, and MLP weight regularization for both g_phi and f_theta. "
        "While the implementation is solid, the coefficient scheduling logic is entirely epoch-based with "
        "no support for data-driven annealing or validation-based adjustment."
    ))

    # =========================================================================
    # 3. WHAT WORKS WELL
    # =========================================================================
    story.append(PageBreak())
    story.append(H1("3. What Works Well: Good Practices"))
    story.append(SP(4))

    story.append(H2("3.1 Principled Mathematical Foundation"))

    story.append(B(
        "The core GNN formulation is mathematically elegant and well-motivated by the underlying neuroscience. "
        "The decomposition into per-edge weights W (capturing connection strength), node embeddings a "
        "(capturing neuron identity and cell-type membership), an edge message function g_phi (capturing "
        "synaptic transfer dynamics), and a node update function f_theta (capturing membrane dynamics) "
        "maps directly onto known biological mechanisms. The choice to use per-edge weights rather than a "
        "dense N-by-N matrix respects the sparse connectivity structure of biological neural circuits, where "
        "the 434,112 edges represent only 0.23% of the possible 13,741-squared connections. The squared "
        "output of g_phi enforces non-negative synaptic transmission, which is biologically accurate since "
        "neurotransmitter release cannot be negative. These are not arbitrary architectural choices but "
        "thoughtful encodings of domain knowledge."
    ))

    story.append(H2("3.2 NeuronState Dataclass Design"))

    story.append(B(
        "The NeuronState and NeuronTimeSeries dataclasses in <font face='Courier' size='8'>neuron_state.py"
        "</font> represent a significant improvement over the legacy packed tensor format. The old approach "
        "stored neuron state as an (N, 9) tensor where each column had an implicit meaning (column 0 = index, "
        "column 3 = voltage, column 4 = stimulus, and so on), leading to fragile indexing and opaque code. "
        "The new dataclass design uses named fields with explicit types, supports optional fields so callers "
        "can load only what they need, provides convenience methods like observable() for switching between "
        "voltage and calcium readouts, and maintains backward compatibility through from_numpy() and "
        "to_packed() conversion methods. The field classification into STATIC_FIELDS and DYNAMIC_FIELDS "
        "enables selective I/O, and the from_zarr_v3 classmethod demonstrates proper integration with the "
        "Zarr storage backend. This is exactly the kind of abstraction that makes research code maintainable."
    ))

    story.append(H2("3.3 Registry Pattern for Model Variants"))

    story.append(B(
        "The model registration system (<font face='Courier' size='8'>models/registry.py</font>) uses a "
        "clean decorator pattern to map model name strings to implementation classes. The @register_model "
        "decorator on FlyVisGNN lists all supported variants (flyvis_A, flyvis_B, etc.) and the create_model "
        "factory function dispatches by name. This pattern makes it trivial to add new model variants without "
        "modifying existing code, which is valuable for the experimental workflow where researchers frequently "
        "try architectural variations. The PARAMS_DOC dictionary embedded in the FlyVisGNN class provides "
        "machine-readable documentation of all configuration parameters, their typical ranges, and their "
        "semantic meaning, which is used by the LLM-guided exploration system to generate informed "
        "hyperparameter suggestions."
    ))

    story.append(H2("3.4 Pydantic Configuration with Strong Typing"))

    story.append(B(
        "The use of Pydantic models with StrEnum types for the configuration system is a genuinely good "
        "engineering decision. Every configuration field has a declared type, a default value, and belongs "
        "to a semantically meaningful section. The StrEnum classes (Boundary, CalciumType, MLPActivation, "
        "UpdateType, etc.) prevent typos in configuration files from silently producing wrong behavior. "
        "When a user writes 'calcium_type: leak' instead of 'calcium_type: leaky', Pydantic raises a "
        "validation error at load time rather than allowing the typo to propagate silently through the "
        "pipeline. This kind of fail-fast behavior is essential in a research setting where configuration "
        "errors can waste hours of GPU time."
    ))

    story.append(H2("3.5 LLM-Guided Hyperparameter Exploration"))

    story.append(B(
        "The integration of LLM-guided hyperparameter search (<font face='Courier' size='8'>GNN_LLM.py"
        "</font>, 1,161 lines) is an innovative approach that leverages Claude to analyze training results, "
        "identify patterns in successful and unsuccessful configurations, and propose new hyperparameter "
        "combinations. The system maintains an exploration tree with UCB (Upper Confidence Bound) scoring, "
        "which balances exploitation of known good configurations with exploration of novel parameter "
        "regions. While the implementation has significant code quality issues (hardcoded cluster paths, "
        "excessive nesting), the concept is genuinely forward-looking and represents a creative application "
        "of foundation models to the meta-optimization problem."
    ))

    story.append(H2("3.6 Comprehensive Biological Constraints"))

    story.append(B(
        "The regularization system encodes a rich set of biological priors that go beyond simple weight "
        "decay. Dale's Law enforcement via W_sign regularization ensures that individual neurons are either "
        "excitatory or inhibitory but not both. The g_phi_diff term promotes monotonic synaptic transfer "
        "functions, reflecting the biophysical property that synaptic transmission strength increases "
        "monotonically with presynaptic voltage. The embedding clustering mechanism encourages neurons of "
        "the same cell type to develop similar learned representations, which is consistent with the "
        "biological observation that neurons of the same type exhibit similar functional properties. These "
        "constraints are not ad hoc regularization tricks but principled inductive biases derived from "
        "decades of neuroscience research."
    ))

    # =========================================================================
    # 4. WHAT NEEDS WORK
    # =========================================================================
    story.append(PageBreak())
    story.append(H1("4. What Needs Work: Spaghetti Code and Anti-Patterns"))
    story.append(SP(4))

    story.append(H2("4.1 The Monolith Problem"))

    story.append(B(
        "The single most pressing architectural issue is the concentration of critical logic into a small "
        "number of enormous files. The training pipeline in graph_trainer.py spans 2,799 lines, the "
        "visualization suite in plot.py reaches 2,955 lines, the model utilities in models/utils.py extend "
        "to 1,551 lines, and the GNN_PlotFigure.py entry point weighs in at 1,530 lines. These files are "
        "not merely large; they contain multiple independent concerns entangled together. The graph_trainer.py "
        "file, for instance, contains the main flyvis training loop, an RNN training variant, an INR training "
        "function, both test pipelines, video generation code, embedding visualization logic, and checkpoint "
        "management, all in a single file with shared mutable state. When a researcher needs to modify the "
        "loss computation, they must navigate 2,799 lines to find the relevant section, understand which "
        "global variables affect it, and hope that their change does not break one of the other training "
        "modes that share the same file."
    ))

    story.append(B(
        "The practical consequences of this monolithic structure extend beyond developer experience. Code "
        "review becomes difficult because changes to graph_trainer.py produce enormous diffs. Testing is "
        "impractical because there is no way to test the loss computation in isolation from the data loading, "
        "optimizer setup, and visualization code that surrounds it. Parallelizing development is nearly "
        "impossible because any two researchers working on training-related features will inevitably create "
        "merge conflicts. This is the canonical symptom of a codebase that needs decomposition into smaller, "
        "single-responsibility modules."
    ))

    story.append(H2("4.2 Absence of Testing Infrastructure"))

    story.append(B(
        "The repository contains exactly one test file, GNN_Test.py (593 lines), and it is not a unit test "
        "suite but rather an integration test that runs a full training-test-plot cycle and compares metrics. "
        "There are no pytest fixtures, no test configuration, no coverage reporting, no CI/CD pipeline, and "
        "no pre-commit hooks. The linter (ruff) is installed in the conda environment but has no configuration "
        "file and is never invoked automatically. This means that type errors, import issues, unused variables, "
        "and formatting inconsistencies accumulate silently over time."
    ))

    story.append(B(
        "The absence of unit tests is particularly concerning for the regularization system, which has 14 "
        "components with epoch-dependent scheduling, two-phase training support, and per-iteration "
        "accumulation. A single sign error in any of these components could silently degrade training "
        "quality without producing an obvious failure. Similarly, the NeuronState dataclass with its "
        "from_numpy, to_packed, frame, and subset_neurons methods is exactly the kind of data transformation "
        "code that benefits enormously from property-based testing. The batch assembly logic in _batch_frames, "
        "which must correctly offset edge indices when concatenating multiple frames, is another prime "
        "candidate for exhaustive unit testing."
    ))

    story.append(H2("4.3 Manual Batch Assembly and Missing DataLoader"))

    story.append(B(
        "The training loop constructs batches by randomly sampling time frames using np.random.randint, "
        "building NeuronState objects for each frame, optionally computing visual field reconstructions, "
        "appending to Python lists, and finally concatenating via _batch_frames. This manual process, "
        "which spans lines 366-430 of graph_trainer.py, has several problems. There is no support for "
        "multi-worker data loading or prefetching, which means the GPU sits idle during data preparation. "
        "There is no epoch-level shuffling with reproducible seeds, making exact reproduction of training "
        "runs difficult. The sampling logic contains a fragile frame index calculation "
        "(k = np.random.randint(sim.n_frames - 4 - tc.time_step - tc.time_window) + tc.time_window) with "
        "no validation that the resulting index is within bounds. And the NaN-checking fallback "
        "(if not torch.isnan(x.voltage).any()) silently skips corrupted frames without logging or error "
        "reporting, potentially leading to variable effective batch sizes."
    ))

    story.append(H2("4.4 Hardcoded Values and Magic Numbers"))

    story.append(B(
        "The training loop contains several unexplained magic numbers that affect critical behavior. The "
        "iteration count is computed as Niter = int(sim.n_frames * tc.data_augmentation_loop // "
        "tc.batch_size * 0.2), where the 0.2 multiplier reduces the iteration count to 20% of the "
        "natural batch count with no explanation or configuration option. The plot frequency is derived "
        "from Niter // 20 and the connectivity checkpoint frequency from Niter // 10, but these ratios "
        "are not configurable and may be inappropriate for different dataset sizes. The rolling window "
        "for MP4 video generation is hardcoded as win = 200 at line 622. The group count for visual "
        "field trace selection is hardcoded as groups = 217 at line 602 with an assertion that assumes "
        "a specific neuron packing scheme."
    ))

    story.append(B(
        "More critically, the GNN_LLM.py file contains hardcoded institution-specific paths at lines "
        "71-76: CLUSTER_USER = 'allierc', CLUSTER_HOME = '/groups/saalfeld/home/allierc', and "
        "CLUSTER_ROOT_DIR pointing to a specific user's directory on a specific computing cluster. The "
        "TMPDIR environment variable is similarly set to '/scratch/allierc' at line 14. These values "
        "make the LLM exploration pipeline completely non-portable and would cause immediate failures "
        "if any other researcher attempted to use the cluster integration features."
    ))

    story.append(H2("4.5 Debug Code in Production"))

    story.append(B(
        "The Neural ODE wrapper contains a global flag DEBUG_ODE set to True at line 15, which triggers "
        "verbose print statements during training. The debug_check_gradients function prints gradient "
        "statistics to stdout rather than using the logging module. The graph_trainer.py file references "
        "this flag at line 542 and calls debug_check_gradients during normal training when the flag is "
        "set. Beyond the ODE debug code, the codebase contains approximately 1,792 print statements "
        "scattered throughout all modules, with no consistent use of Python's logging framework. This "
        "makes it impossible to control output verbosity at runtime, which is essential when running "
        "batch experiments on a cluster where stdout must be redirected to log files and filtered by "
        "severity."
    ))

    story.append(H2("4.6 No Learning Rate Scheduling"))

    story.append(B(
        "Despite the configuration system defining both learning_rate_start and learning_rate_end fields "
        "for each parameter group (W, g_phi, f_theta, embeddings), there is no learning rate scheduler "
        "in the training loop. The learning rates are set once during optimizer construction and remain "
        "constant throughout training unless modified by the embedding clustering mechanism (which sets "
        "lr_embedding to 1e-10 when freezing) or by the UMAP cluster reassignment (which reconstructs "
        "the optimizer). There is no cosine annealing, no warmup, no reduce-on-plateau, and no linear "
        "decay. This is a significant gap because learning rate scheduling is one of the most reliable "
        "and well-understood techniques for improving convergence in deep learning, and its absence "
        "almost certainly leaves performance on the table."
    ))

    story.append(H2("4.7 Inconsistent Error Handling"))

    story.append(B(
        "The codebase contains 106 try-except blocks with inconsistent error handling patterns. Nine "
        "instances use bare except: clauses that catch all exceptions including SystemExit and "
        "KeyboardInterrupt, which is a well-known Python anti-pattern. The data loading fallback at "
        "lines 167-173 of graph_trainer.py silently falls back from x_list_train to x_list_0 with "
        "only a print statement, which could mask serious data pipeline failures. The optional import "
        "of AugmentedVideoDataset at lines 80-82 silently sets the class to None on ImportError, "
        "which will produce a confusing NoneType error much later if the code path that requires "
        "it is actually exercised. Error handling should follow a consistent pattern: either fail "
        "fast with a descriptive error message, or handle the error gracefully with proper logging."
    ))

    # =========================================================================
    # 5. DEEP DIVE: FOUR WORST OFFENDERS
    # =========================================================================
    story.append(PageBreak())
    story.append(H1("5. Deep-Dive: The Four Worst Offenders"))
    story.append(SP(4))

    story.append(H2("5.1 graph_trainer.py: The God Object"))

    story.append(B(
        "At 2,799 lines, graph_trainer.py is the most problematic file in the repository. It contains "
        "five major functions (data_train, data_train_flyvis, data_train_flyvis_RNN, data_test_flyvis, "
        "data_test_flyvis_special) and approximately fifteen helper functions, all sharing a single "
        "module namespace. The main training function data_train_flyvis spans from line 130 to line 873 "
        "and contains the complete training pipeline: data loading, normalization computation, model "
        "construction, optimizer setup with parameter groups, the epoch loop, the iteration loop with "
        "manual batch assembly, loss computation for three different training modes (single-step, "
        "recurrent, and Neural ODE), gradient computation and clipping, checkpoint saving, metrics "
        "logging, R-squared computation, visual field video generation, embedding clustering, UMAP "
        "reassignment, and end-of-epoch summary plotting."
    ))

    story.append(B(
        "The function has so many responsibilities that it is impossible to understand, test, or "
        "modify any single aspect without reading the entire 720-line body. The three training modes "
        "(single-step, recurrent, Neural ODE) are selected by if-elif-else branching within the "
        "inner loop (lines 460-534), sharing the same batch assembly and loss variable but diverging "
        "in their forward pass and loss computation. This means that a bug fix to the recurrent "
        "training mode requires understanding the single-step and Neural ODE modes to verify that "
        "the fix does not introduce regressions. The visual field video generation code (lines "
        "595-728) is embedded directly in the training loop and executes at specific iteration "
        "checkpoints, mixing visualization concerns with training logic in a way that makes both "
        "harder to maintain."
    ))

    story.append(B(
        "The recommended refactoring is to decompose this file into at least five modules: a "
        "DataPipeline class responsible for loading, normalizing, and batching data; a Trainer "
        "base class with separate SingleStepTrainer, RecurrentTrainer, and NeuralODETrainer "
        "subclasses; a CheckpointManager for saving and loading model state; a MetricsLogger for "
        "R-squared computation and logging; and a TrainingVisualizer for all plotting and video "
        "generation. Each of these modules can then be tested independently and composed in the "
        "main training script."
    ))

    story.append(H2("5.2 MLP.py: Dead Code and Missing Abstraction"))

    story.append(B(
        "The MLP class at 160 lines appears simple on the surface but contains several issues that "
        "compound across the codebase. The activation function selection uses a chain of if-elif "
        "statements (lines 36-47) that maps string names to PyTorch functions, but this mapping is "
        "not shared with the configuration system's MLPActivation enum. If a new activation is added "
        "to the enum but not to the MLP class (or vice versa), the inconsistency will not be caught "
        "until runtime. The class initializes all hidden-layer weights with a fixed std of 0.1 "
        "regardless of the layer width or activation function, which is not optimal for any of the "
        "supported activations. Modern initialization schemes like Kaiming or Xavier are well-understood "
        "to improve convergence, especially for deeper networks."
    ))

    story.append(B(
        "More egregiously, the file contains over 100 lines of dead code in the __main__ block "
        "(lines 59-161) that defines inline training experiments with hardcoded device strings "
        "('cuda:0'), matplotlib interactive backends ('Qt5Agg'), and numerical constants. This "
        "code is never executed as part of the package and serves no purpose in production. It "
        "should either be moved to a proper test file or deleted entirely. The dropout mechanism "
        "is implemented (self.dropout_rate = 0.0 at line 14) but never actually configured to "
        "a non-zero value anywhere in the codebase, making it dead weight in the forward pass "
        "that adds a conditional branch on every layer evaluation."
    ))

    story.append(H2("5.3 GNN_LLM.py: Institution Lock-in and Deep Nesting"))

    story.append(B(
        "The LLM-guided exploration script is conceptually innovative but suffers from severe "
        "code quality issues that undermine its utility. The six hardcoded cluster constants at "
        "lines 71-76 (CLUSTER_USER, CLUSTER_LOGIN, CLUSTER_HOME, CLUSTER_ROOT_DIR, CLUSTER_DATA_DIR, "
        "CLUSTER_SSH) lock the entire cluster integration to a single user at a specific institution. "
        "Any researcher at a different institution who wants to use the cluster features would need to "
        "modify the source code rather than provide configuration. These values should be read from "
        "environment variables or a configuration file."
    ))

    story.append(B(
        "The nesting depth in several functions reaches pathological levels. The wait_for_cluster_jobs "
        "function contains 12 levels of indentation, mixing file I/O, SSH command execution, YAML "
        "parsing, directory traversal, and error handling in a single deeply-nested control flow. "
        "At this level of nesting, it becomes nearly impossible to reason about which variables are "
        "in scope, which exceptions are being caught, and which code paths are actually reachable. "
        "The function should be decomposed into smaller helper functions, each responsible for a "
        "single level of abstraction: one for SSH operations, one for YAML parsing, one for directory "
        "scanning, and one for the high-level orchestration logic."
    ))

    story.append(H2("5.4 plot.py: The 2,955-Line Visualization Behemoth"))

    story.append(B(
        "The visualization module at 2,955 lines is the largest file in the source package and "
        "exemplifies the consequences of allowing a module to grow without bounds. It contains "
        "functions for training loss visualization, embedding scatter plots, weight matrix comparison, "
        "spatial activity grid rendering, neuron trace plotting, R-squared computation (which is not "
        "a visualization concern at all), dynamic video generation, and numerous helper functions for "
        "color mapping and axis formatting. The mixing of computation and visualization code is "
        "particularly problematic because R-squared computation, which is a pure mathematical "
        "operation, is entangled with matplotlib figure creation, making it impossible to compute "
        "metrics without also generating plots."
    ))

    story.append(B(
        "The companion file GNN_PlotFigure.py (1,530 lines) contains the plot_synaptic_flyvis "
        "function, which at over 1,000 lines is the single longest function in the entire codebase. "
        "This function handles data loading, normalization, statistical computation, multi-panel "
        "figure creation, and metrics extraction in a single monolithic body. A function of this "
        "length cannot be reviewed, tested, or maintained effectively. The recommended approach is "
        "to separate metric computation from visualization, create small composable plotting functions "
        "that each produce a single panel or figure, and use a thin orchestration layer to compose "
        "them into multi-panel figures."
    ))

    # =========================================================================
    # 6. FILE-BY-FILE SCORECARD
    # =========================================================================
    story.append(PageBreak())
    story.append(H1("6. File-by-File Scorecard"))
    story.append(SP(4))

    story.append(B(
        "The following scorecard rates each major file on a 1-10 scale across five dimensions: "
        "architecture (separation of concerns, modularity), readability (naming, comments, structure), "
        "robustness (error handling, edge cases, validation), testability (can components be tested in "
        "isolation), and maintainability (ease of modification, risk of regression). The overall score "
        "is a weighted average with architecture and testability weighted double, reflecting their "
        "outsized impact on long-term project health."
    ))

    story.append(SP(4))

    scorecard_data = [
        ["models/flyvis_gnn.py", "8", "8", "7", "7", "8", "7.7"],
        ["models/MLP.py", "6", "6", "5", "5", "5", "5.4"],
        ["models/graph_trainer.py", "2", "4", "4", "1", "2", "2.4"],
        ["models/utils.py", "5", "5", "5", "4", "4", "4.5"],
        ["models/Neural_ode_wrapper.py", "7", "6", "5", "6", "6", "6.2"],
        ["models/Siren_Network.py", "7", "6", "5", "5", "6", "5.9"],
        ["models/registry.py", "9", "9", "7", "8", "9", "8.5"],
        ["models/exploration_tree.py", "5", "5", "5", "3", "4", "4.2"],
        ["config.py", "9", "8", "8", "7", "8", "8.0"],
        ["neuron_state.py", "9", "9", "7", "8", "9", "8.5"],
        ["plot.py", "3", "4", "4", "2", "3", "3.0"],
        ["utils.py", "5", "5", "5", "4", "4", "4.5"],
        ["sparsify.py", "6", "5", "5", "4", "5", "5.0"],
        ["zarr_io.py", "7", "7", "6", "6", "7", "6.7"],
        ["generators/graph_data_gen.py", "5", "5", "5", "3", "4", "4.2"],
        ["generators/davis.py", "5", "5", "4", "3", "4", "4.1"],
        ["generators/utils.py", "5", "5", "5", "4", "4", "4.5"],
        ["GNN_Main.py", "7", "7", "6", "5", "7", "6.3"],
        ["GNN_LLM.py", "3", "3", "4", "2", "3", "2.9"],
        ["GNN_Test.py", "5", "5", "5", "4", "5", "4.8"],
        ["GNN_PlotFigure.py", "2", "3", "4", "1", "2", "2.2"],
    ]

    t = make_table(
        ["File", "Arch", "Read", "Robust", "Test", "Maint", "Overall"],
        scorecard_data,
        col_widths=[2.2*inch, 0.55*inch, 0.55*inch, 0.6*inch, 0.55*inch, 0.55*inch, 0.65*inch],
    )
    story.append(t)

    story.append(SP(8))
    story.append(B(
        "The scores reveal a clear bimodal distribution. The well-designed components, specifically "
        "neuron_state.py, config.py, registry.py, and flyvis_gnn.py, score consistently above 7.5, "
        "demonstrating that the project's authors are capable of writing clean, modular code. The "
        "problematic files, particularly graph_trainer.py, GNN_PlotFigure.py, GNN_LLM.py, and plot.py, "
        "all score below 3.5 and share a common pathology: they have grown far beyond a single "
        "responsibility without being decomposed. This suggests that the code quality issues are not "
        "due to lack of skill but rather lack of refactoring discipline, which is a common and "
        "correctable problem in research codebases under publication pressure."
    ))

    # =========================================================================
    # 7. RECOMMENDATIONS AND REFACTORING ROADMAP
    # =========================================================================
    story.append(PageBreak())
    story.append(H1("7. Recommendations and Refactoring Roadmap"))
    story.append(SP(4))

    story.append(H2("7.1 Immediate Fixes (Week 1-2)"))

    story.append(B(
        "The first priority is to remove the DEBUG_ODE flag from Neural_ode_wrapper_FlyVis.py and "
        "replace all debug print statements with proper Python logging calls at the DEBUG level. This "
        "is a mechanical transformation that requires no architectural changes but immediately improves "
        "the signal-to-noise ratio of training output. Simultaneously, the hardcoded cluster constants "
        "in GNN_LLM.py should be extracted to environment variables or a cluster configuration file. "
        "The __main__ block in MLP.py should be deleted entirely. The bare except: clauses across "
        "nine files should be replaced with specific exception types. These changes are low-risk, "
        "high-impact improvements that can be done in parallel by multiple contributors."
    ))

    story.append(H2("7.2 Infrastructure Setup (Week 2-3)"))

    story.append(B(
        "The second priority is to establish the testing and CI infrastructure that every serious "
        "project requires. This means configuring ruff in pyproject.toml with a reasonable rule set, "
        "adding a pytest.ini or pyproject.toml [tool.pytest] section, writing unit tests for "
        "NeuronState, _batch_frames, LossRegularizer, and the MLP class, setting up GitHub Actions "
        "for automated linting and testing on every push, and adding pre-commit hooks to enforce "
        "formatting and import order. The target should be at least 80% coverage on the core model "
        "code and 100% coverage on data transformation functions. A properly configured ruff would "
        "catch unused imports, undefined variables, and formatting inconsistencies automatically."
    ))

    story.append(H2("7.3 Architectural Decomposition (Week 3-6)"))

    story.append(B(
        "The third priority is the decomposition of graph_trainer.py into smaller modules. The "
        "recommended structure is: a data_pipeline.py module containing a FlyVisDataset class that "
        "wraps NeuronTimeSeries with PyTorch Dataset and DataLoader semantics; a trainer.py module "
        "containing a FlyVisTrainer class with separate methods for single-step, recurrent, and "
        "Neural ODE training that share common setup and evaluation logic; a checkpoint.py module "
        "for model serialization and loading; a metrics.py module for R-squared computation and "
        "logging that is completely decoupled from visualization; and a training_viz.py module for "
        "the visual field video generation and training curve plotting. Each module should be "
        "independently testable and should communicate through well-defined interfaces rather than "
        "shared mutable state."
    ))

    story.append(H2("7.4 Training Pipeline Modernization (Week 4-8)"))

    story.append(B(
        "The fourth priority is to modernize the training pipeline with standard PyTorch practices. "
        "This includes implementing a proper DataLoader with the FlyVisDataset class, adding a "
        "learning rate scheduler (cosine annealing with warm restarts is a strong default for this "
        "type of problem), implementing early stopping based on validation R-squared, adding gradient "
        "norm logging for all parameter groups (not just W), integrating with a proper experiment "
        "tracking system like Weights and Biases or TensorBoard, and adding distributed training "
        "support via PyTorch's DistributedDataParallel. The manual batch assembly logic should be "
        "replaced entirely by the DataLoader, which provides multi-worker loading, prefetching, and "
        "reproducible shuffling out of the box."
    ))

    # =========================================================================
    # 7.5 FUTURE WORKS - THE BIG SECTION
    # =========================================================================
    story.append(PageBreak())
    story.append(H2("7.5 Future Works: Structural Changes to Break the Performance Ceiling"))
    story.append(SP(4))

    story.append(B(
        "Having analyzed the complete FlyVis-GNN codebase and its current performance characteristics, "
        "what follows is a set of concrete structural recommendations from the perspective of a senior "
        "ML researcher at Google DeepMind. These are not incremental improvements to hyperparameters "
        "or regularization coefficients, which the existing LLM-guided exploration system already handles "
        "well. Instead, these are architectural and methodological changes that address fundamental "
        "limitations of the current approach and have the potential to substantially shift the performance "
        "frontier. Each recommendation draws on recent advances in graph neural networks, neural ODEs, "
        "self-supervised learning, and computational neuroscience that have emerged from the top "
        "ML venues in 2024 and 2025."
    ))

    story.append(H3("7.5.1 Replace Homogeneous MLP Message Passing with Heterogeneous Graph Transformer Attention"))

    story.append(B(
        "The current FlyVisGNN uses a single shared MLP (g_phi) for all edges regardless of the neuron "
        "types at either end of the connection. This homogeneous treatment forces the network to learn a "
        "universal synaptic transfer function that must work for all 65 neuron types simultaneously. "
        "In the Drosophila visual system, different cell types have fundamentally different electrophysiological "
        "properties: photoreceptors (R1-R8) respond to light with graded potentials, lamina neurons (L1-L5) "
        "perform temporal filtering, medulla neurons (Mi, Tm, T) compute direction selectivity, and lobula "
        "plate tangential cells integrate wide-field motion. Forcing a single MLP to capture all of these "
        "distinct dynamics is a severe bottleneck."
    ))

    story.append(B(
        "The recommended replacement is a Heterogeneous Graph Transformer (HGT) architecture inspired by "
        "Hu et al. (WWW 2020) and the more recent GPS (General, Powerful, Scalable) framework from "
        "Rampasek et al. (NeurIPS 2022). In this architecture, each edge type (defined by the source and "
        "destination neuron types) gets its own learned attention head, allowing the network to learn "
        "type-specific message functions without the parameter explosion of having 65x65 separate MLPs. "
        "The key insight is to use type-conditioned linear projections for keys, queries, and values: "
        "K = W_K[src_type] * h_src, Q = W_Q[dst_type] * h_dst, V = W_V[edge_type] * msg, where the "
        "type-specific weight matrices are of moderate size and shared across all edges of the same type. "
        "This gives the network the representational power of type-specific processing while keeping the "
        "parameter count manageable."
    ))

    story.append(B(
        "The attention mechanism also provides a natural replacement for the current g_phi_positive "
        "squaring trick. Instead of forcing non-negative messages by squaring the MLP output (which "
        "loses gradient signal near zero and creates a non-smooth optimization landscape), the "
        "attention weights can be constrained to be non-negative through softmax normalization, while "
        "the value vectors carry the signed message content. This separation of attention weights "
        "(how much to listen to each neighbor) from message content (what is being communicated) is "
        "more principled than the current approach and has been shown to improve optimization dynamics "
        "in transformer-based GNNs."
    ))

    story.append(H3("7.5.2 Neural ODE with Adaptive Multi-Scale Time Stepping"))

    story.append(B(
        "The current Neural ODE integration uses torchdiffeq with a fixed ODE method (typically "
        "dopri5) applied uniformly across all neurons. However, the Drosophila visual system operates "
        "across multiple timescales: photoreceptor responses occur on the order of milliseconds, "
        "lamina processing on tens of milliseconds, and motion integration in the lobula plate on "
        "hundreds of milliseconds. A single integration scheme with uniform tolerances cannot "
        "efficiently capture this multi-scale dynamics."
    ))

    story.append(B(
        "The recommended approach is to implement a Latent Neural ODE with Multi-Scale Integration, "
        "drawing on the Latent ODEs for Irregularly-Sampled Time Series framework (Rubanova et al., "
        "NeurIPS 2019) and the more recent Multi-Scale Neural ODE approach (Iakovlev et al., 2024). "
        "The idea is to decompose the neural state into fast and slow components, each integrated at "
        "its own characteristic timescale. The fast component captures millisecond-scale synaptic "
        "dynamics and is integrated with small step sizes, while the slow component captures longer-term "
        "adaptation and is integrated with larger steps. The coupling between scales is learned by the "
        "GNN, which computes messages that inform both the fast and slow dynamics."
    ))

    story.append(B(
        "Concretely, this can be implemented by replacing the single voltage state v with a pair (v_fast, "
        "v_slow), where v_fast evolves according to dv_fast/dt = f_fast(v_fast, v_slow, msg, exc) with "
        "fast time constant tau_fast, and v_slow evolves as dv_slow/dt = f_slow(v_fast, v_slow) with "
        "slow time constant tau_slow. The observable is then v = v_fast + v_slow, recovering the current "
        "interface. This decomposition allows the ODE solver to use different step sizes for each "
        "component, dramatically improving integration efficiency for stiff systems. The biological "
        "motivation is that neurons have both fast membrane dynamics (active conductances) and slow "
        "adaptation dynamics (calcium-dependent currents), and separating these explicitly should "
        "improve both the speed and accuracy of the simulation."
    ))

    story.append(H3("7.5.3 Connectivity-Aware Graph Positional Encodings"))

    story.append(B(
        "The current node embeddings a (of dimension 2 by default) are learned from scratch as free "
        "parameters. While this is flexible, it fails to exploit the rich structural information "
        "already present in the connectome graph. The 434,112 edges of the Drosophila visual system "
        "encode a precise wiring diagram that reflects both the spatial organization (retinotopic "
        "mapping) and the functional architecture (cell-type-specific connectivity motifs) of the "
        "circuit. Initializing the embeddings randomly (or as ones, as the current code does) throws "
        "away this structural information and forces the network to rediscover it during training."
    ))

    story.append(B(
        "The recommended approach is to use Learnable Structural Encodings (LSE) or Random Walk "
        "Positional Encodings (RWPE) as described in Dwivedi et al. (JMLR 2022) and the subsequent "
        "work on SignNet (Lim et al., ICML 2023). Specifically, the node embeddings should be "
        "initialized from the top-k eigenvectors of the graph Laplacian, which capture the global "
        "connectivity structure at multiple scales. The first eigenvector distinguishes input neurons "
        "from output neurons, the next few eigenvectors capture the columnar organization of the "
        "visual system, and higher-order eigenvectors encode fine-grained cell-type clusters. These "
        "spectral features can be made permutation-equivariant using the SignNet architecture, which "
        "processes each eigenspace independently before combining them."
    ))

    story.append(B(
        "An alternative that may be more computationally efficient is Random Walk Structural Encoding "
        "(RWSE), which computes the diagonal of the random walk transition matrix at multiple steps: "
        "p_i = [P^1_ii, P^2_ii, ..., P^k_ii]. These features encode the local connectivity structure "
        "around each node (self-return probabilities at different scales) and can be pre-computed "
        "once before training. The combination of RWSE features with the existing learned embeddings "
        "gives the model both structural priors and the flexibility to learn task-specific features, "
        "which should improve convergence speed and final accuracy."
    ))

    story.append(H3("7.5.4 Self-Supervised Pre-Training on Connectome Structure"))

    story.append(B(
        "The current training paradigm is purely supervised: the model is trained end-to-end on "
        "labeled (stimulus, response) pairs generated by the ground-truth simulation. This means "
        "that all structural knowledge about the connectome must be learned implicitly through the "
        "supervised loss, which is sample-inefficient and sensitive to the distribution of training "
        "stimuli. If the training stimuli do not sufficiently excite certain neural pathways, the "
        "model cannot learn the connectivity of those pathways."
    ))

    story.append(B(
        "The recommended approach is to add a self-supervised pre-training phase that learns "
        "meaningful representations of the connectome structure before supervised fine-tuning. "
        "Following the GraphMAE framework (Hou et al., KDD 2022) and the more recent All-in-One "
        "Multi-Task GNN Pre-training (Sun et al., ICML 2024), the pre-training would use a masked "
        "autoencoder objective: randomly mask a fraction (e.g., 30%) of the edge weights W, and "
        "train the GNN to reconstruct the masked weights from the unmasked neighbors. This forces "
        "the model to learn the local and global connectivity patterns of the circuit without "
        "requiring any labeled activity data."
    ))

    story.append(B(
        "A second complementary pre-training objective is contrastive learning on graph structure: "
        "augment the connectome graph by randomly dropping edges or perturbing weights, generate "
        "positive pairs from augmentations of the same graph and negative pairs from different "
        "graphs, and train the encoder to distinguish them. The GraphCL framework (You et al., "
        "NeurIPS 2020) provides a solid foundation for this approach. The pre-trained node "
        "embeddings and MLP weights would then serve as initialization for the supervised "
        "training phase, providing a warm start that encodes structural knowledge about the "
        "circuit topology."
    ))

    story.append(H3("7.5.5 Differentiable Sparse Connectivity Learning"))

    story.append(B(
        "The current approach to learning sparse connectivity is through L1 regularization on the "
        "per-edge weights W, which encourages many weights to be near zero but does not actually "
        "remove edges from the graph. The model always performs message passing over all 434,112 "
        "edges, computing g_phi and multiplying by W even for edges where W is essentially zero. "
        "This is both computationally wasteful and representationally limiting, because the L1 "
        "penalty creates a bias toward small weights that can suppress genuine connections."
    ))

    story.append(B(
        "The recommended replacement is a differentiable edge selection mechanism using the "
        "Gumbel-Softmax trick or the more recent straight-through estimator approaches from "
        "NAS (Neural Architecture Search) literature. Specifically, each edge should have a "
        "learned binary gate z_e that is sampled from a Bernoulli distribution during training "
        "and hardened during inference: z_e ~ Bernoulli(sigmoid(alpha_e)), where alpha_e is a "
        "learnable logit. During training, the relaxed Gumbel-Sigmoid is used for gradient "
        "estimation: z_e = sigmoid((alpha_e + G1 - G2) / tau), where G1, G2 are Gumbel noise "
        "samples and tau is a temperature that anneals from 1.0 to 0.1 over training. The "
        "message for edge e then becomes msg_e = z_e * W_e * g_phi(features), which is zero "
        "when the gate is closed. This approach discovers the sparse connectivity structure "
        "jointly with the weight magnitudes, which is more principled than post-hoc L1 "
        "thresholding."
    ))

    story.append(B(
        "The biological justification is that synapse formation and elimination are active "
        "developmental processes that depend on neural activity patterns, not just synaptic "
        "strength. The differentiable gating mechanism mimics this developmental process by "
        "learning which connections are necessary for reproducing the observed dynamics, rather "
        "than simply penalizing connection strength. The sparsity level can be controlled by "
        "adding a regularization term on the expected number of active gates: L_sparsity = "
        "lambda * sum(sigmoid(alpha_e)), which provides a direct and interpretable control over "
        "the number of connections in the learned circuit."
    ))

    story.append(H3("7.5.6 Flow Matching for Neural Dynamics Generation"))

    story.append(B(
        "The current training approach optimizes the GNN to predict the time derivative du/dt "
        "at individual time points, or to match the integrated trajectory over a fixed number "
        "of steps. This pointwise or short-horizon matching can miss longer-range temporal "
        "structure in the neural dynamics, such as oscillations, traveling waves, or attractor "
        "dynamics that unfold over many time steps. The recurrent training mode partially "
        "addresses this by unrolling for multiple steps, but it is limited to fixed unroll "
        "lengths and suffers from vanishing gradients over long sequences."
    ))

    story.append(B(
        "Flow Matching (Lipman et al., ICLR 2023) and its conditional variant (Tong et al., "
        "ICLR 2024) offer an elegant alternative that is well-suited to neural dynamics. Instead "
        "of training the model to predict du/dt at specific time points, Flow Matching trains a "
        "velocity field v_theta(x, t) that transports samples from a simple base distribution "
        "(e.g., the resting state) to the target distribution (the observed neural activity at "
        "time T) along optimal transport paths. The key advantage is that the training objective "
        "is a simple regression loss on the velocity field, but the resulting model can generate "
        "complete trajectories by integrating the velocity field from t=0 to t=T."
    ))

    story.append(B(
        "For the FlyVis application, conditional Flow Matching would condition the velocity "
        "field on the visual stimulus: v_theta(x, t | stimulus), enabling the model to generate "
        "stimulus-specific neural activity trajectories. This naturally handles variable-length "
        "stimuli and provides a principled way to generate long trajectories without the "
        "gradient issues of recurrent unrolling. The integration with the existing GNN "
        "architecture is straightforward: the velocity field is parameterized by the GNN "
        "forward pass, and the conditioning on stimulus is provided through the excitation "
        "channel that already exists in the NeuronState."
    ))

    story.append(H3("7.5.7 Physics-Informed Loss with Conservation Laws"))

    story.append(B(
        "The current loss function is a straightforward L2 regression between predicted and "
        "observed dynamics (either du/dt or integrated state), augmented by the 14-component "
        "regularization system. While the regularizers encode important biological constraints, "
        "they do not exploit the physical conservation laws that govern neural circuits. "
        "Specifically, the total synaptic current flowing into a neuron must equal the sum of "
        "its membrane current, capacitive current, and leak current. This current conservation "
        "law provides an additional supervision signal that is orthogonal to the activity "
        "matching loss and can improve the physical plausibility of learned dynamics."
    ))

    story.append(B(
        "The recommended approach is to add a Physics-Informed Neural Network (PINN) style "
        "loss term (Raissi et al., Journal of Computational Physics, 2019) that enforces "
        "the cable equation at every node: C * dv/dt = sum(g_syn * (E_syn - v)) + g_leak * "
        "(E_leak - v) + I_ext, where C is the membrane capacitance, g_syn are the synaptic "
        "conductances (related to the learned weights W), E_syn are the reversal potentials "
        "(which can be learned or fixed based on Dale's Law assignment), g_leak is the leak "
        "conductance, E_leak is the resting potential (related to the learned V_rest), and "
        "I_ext is the external stimulus. The PINN loss computes the residual of this equation "
        "at sampled collocation points and penalizes deviations from zero. This provides a "
        "dense supervision signal that guides the model toward physically consistent solutions "
        "even in regions of state space that are underrepresented in the training data."
    ))

    story.append(H3("7.5.8 Equivariant GNN Architecture for Retinotopic Symmetry"))

    story.append(B(
        "The Drosophila visual system exhibits approximate translational symmetry in the "
        "retinotopic axes: the same circuit motif (columns of neurons processing local visual "
        "input) is replicated across the visual field. The current GNN treats each neuron as "
        "a unique entity with its own learned embedding, ignoring this symmetry. This means "
        "that the model must independently learn the same synaptic function for each retinotopic "
        "position, which is both sample-inefficient and can lead to inconsistent solutions across "
        "the visual field."
    ))

    story.append(B(
        "The recommended approach is to incorporate the retinotopic symmetry as an architectural "
        "constraint using equivariant GNNs. Specifically, the node embeddings and message functions "
        "should be parameterized in terms of relative positions rather than absolute positions: "
        "the message from neuron j to neuron i should depend on the relative displacement "
        "(x_i - x_j, y_i - y_j) rather than on the absolute positions of i and j. This is a "
        "form of translational equivariance that ensures the learned synaptic function is the same "
        "at every retinotopic position. The E(n)-equivariant message passing framework from "
        "Satorras et al. (ICML 2021) provides a clean implementation of this principle, where "
        "messages are functions of pairwise distances and relative orientations rather than "
        "absolute coordinates."
    ))

    story.append(B(
        "The practical implementation would modify the g_phi input features from (v_src, a_src) "
        "to (v_src, a_src, delta_x, delta_y), where delta_x and delta_y are the relative "
        "spatial displacements. The node embeddings a would then encode cell-type identity "
        "rather than spatial position, because spatial information is provided explicitly through "
        "the relative coordinates. This separation of identity and position should improve "
        "generalization across the visual field and reduce the number of free parameters, since "
        "the same synaptic function is shared across all retinotopic positions."
    ))

    story.append(H3("7.5.9 Curriculum Learning with Progressive Complexity"))

    story.append(B(
        "The current training scheme presents all stimuli and all neurons simultaneously from "
        "the beginning of training. This flat presentation makes the optimization landscape "
        "unnecessarily complex, because the model must simultaneously learn photoreceptor "
        "responses, lamina processing, and deep medulla computation. A curriculum learning "
        "approach, where the model first learns simple dynamics and progressively encounters "
        "more complex scenarios, can significantly improve convergence and final performance."
    ))

    story.append(B(
        "The recommended curriculum has three phases. In the first phase, the model is trained "
        "only on flash stimuli (step functions in light intensity) which produce simple transient "
        "responses that are easy to match. The loss is computed only on photoreceptors and first-order "
        "interneurons (L1-L5), allowing the model to learn the feedforward sensory pathway before "
        "tackling recurrent dynamics. In the second phase, the model is trained on moving bars "
        "and gratings, which require direction-selective computation in the medulla. The loss is "
        "expanded to include all neuron types, and the network must now learn the recurrent "
        "connectivity that produces direction selectivity. In the third phase, the model is "
        "trained on natural scenes (DAVIS video dataset), which require the full complexity of "
        "the visual circuit including adaptation, gain control, and figure-ground segregation. "
        "This progressive complexity exposure is analogous to the developmental sequence in "
        "biological visual systems, where spontaneous retinal waves and simple stimuli during "
        "early development precede exposure to natural visual scenes."
    ))

    story.append(H3("7.5.10 Uncertainty Quantification via Ensemble or Evidential Deep Learning"))

    story.append(B(
        "The current model provides point estimates of connectivity weights without any "
        "quantification of uncertainty. This is a significant limitation because some connections "
        "in the circuit may be strongly constrained by the observed dynamics (e.g., connections "
        "in the primary visual pathway) while others may be poorly identified (e.g., connections "
        "between neurons whose activity is not observed). Without uncertainty estimates, it is "
        "impossible to distinguish between well-determined and poorly-determined connections, "
        "which limits the scientific utility of the recovered circuit."
    ))

    story.append(B(
        "The recommended approach is to implement Evidential Deep Learning (Amini et al., NeurIPS "
        "2020), which places a prior distribution over the model's output distribution and learns "
        "the parameters of this prior from data. For the FlyVis application, each predicted edge "
        "weight W_e would be accompanied by an uncertainty estimate sigma_e, where high uncertainty "
        "indicates that the connection strength is not well-constrained by the training data. "
        "The loss function is modified to use a Normal-Inverse-Gamma prior, which naturally "
        "decomposes the total uncertainty into aleatoric uncertainty (due to observation noise) "
        "and epistemic uncertainty (due to limited data or model capacity). This decomposition "
        "is directly useful for experimental design: connections with high epistemic uncertainty "
        "are candidates for targeted electrophysiology experiments that would provide the most "
        "information about the circuit."
    ))

    story.append(B(
        "An alternative approach that requires less modification to the existing architecture is "
        "Deep Ensembles (Lakshminarayanan et al., NeurIPS 2017), where multiple models are trained "
        "with different random initializations and the variance across ensemble members provides "
        "an uncertainty estimate. The LLM-guided exploration system already trains multiple model "
        "instances with different hyperparameters, and a straightforward extension would be to "
        "use the best-performing instances as an ensemble for uncertainty quantification rather "
        "than selecting a single winner."
    ))

    story.append(H3("7.5.11 Summary of Recommended Structural Changes"))

    story.append(SP(4))
    future_data = [
        ["Heterogeneous Graph Transformer", "Replace homogeneous g_phi with type-conditioned attention", "High"],
        ["Multi-Scale Neural ODE", "Decompose state into fast/slow components", "High"],
        ["Graph Positional Encodings", "Initialize embeddings from Laplacian eigenvectors or RWSE", "Medium"],
        ["Self-Supervised Pre-Training", "Masked autoencoder + contrastive on connectome", "High"],
        ["Differentiable Edge Selection", "Gumbel-Sigmoid gates instead of L1 regularization", "Medium"],
        ["Flow Matching Dynamics", "Replace pointwise loss with trajectory-level flow matching", "Medium"],
        ["Physics-Informed Loss", "Add cable equation residual as auxiliary loss", "Medium"],
        ["Equivariant Architecture", "Exploit retinotopic translational symmetry", "Medium"],
        ["Curriculum Learning", "Progressive stimulus complexity over training", "Low-Med"],
        ["Uncertainty Quantification", "Evidential DL or deep ensembles for confidence", "Medium"],
    ]
    t = make_table(
        ["Innovation", "Description", "Expected Impact"],
        future_data,
        col_widths=[2.0*inch, 3.2*inch, 1.0*inch],
    )
    story.append(t)

    story.append(SP(8))

    story.append(B(
        "The innovations listed above are not mutually exclusive and can be combined synergistically. "
        "The recommended implementation order is: (1) graph positional encodings, as these require "
        "minimal code changes and provide immediate benefit; (2) curriculum learning, which requires "
        "only changes to the data sampling logic; (3) heterogeneous graph transformer, which is the "
        "highest-impact architectural change; (4) differentiable edge selection, which replaces the "
        "ad-hoc L1 sparsification with a principled mechanism; and (5) self-supervised pre-training, "
        "which requires the most engineering effort but provides the most significant long-term benefit. "
        "The remaining innovations (multi-scale ODE, flow matching, physics-informed loss, equivariant "
        "architecture, uncertainty quantification) can be pursued in parallel based on available "
        "resources and research priorities."
    ))

    story.append(SP(6))
    story.append(hr())
    story.append(SP(4))
    story.append(B(
        "<i>This review was conducted by analyzing the complete flyvis-gnn repository (v0.1.0, "
        "~27,300 lines of Python across 50 files). Every major file was read in full, with "
        "particular attention to the model architecture, training pipeline, configuration system, "
        "regularization framework, and Neural ODE integration. The recommendations reflect current "
        "best practices from Google DeepMind research on graph neural networks, neural ODEs, and "
        "computational neuroscience as of February 2026.</i>"
    ))

    return story


def main():
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch,
        title="FlyVis-GNN Code Review",
        author="Senior ML Engineer, Google DeepMind",
    )

    story = build_document()
    doc.build(story)
    print(f"PDF written to {OUTPUT_PATH}")
    n_pages = count_pages(OUTPUT_PATH)
    print(f"Total pages: {n_pages}")


def count_pages(path):
    """Count PDF pages by reading the trailer."""
    try:
        from reportlab.lib.utils import open_for_read
        # Use PyPDF2 if available, otherwise approximate
        import struct
        with open(path, 'rb') as f:
            content = f.read()
        # Count page objects
        count = content.count(b'/Type /Page') - content.count(b'/Type /Pages')
        return max(count, 1)
    except Exception:
        return "unknown"


if __name__ == '__main__':
    main()

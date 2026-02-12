# FlyVis-GNN

Graph neural networks for fly visual system connectivity recovery, with LLM-guided exploration.

## Installation

```bash
conda env create -f envs/environment.linux.yaml
conda activate flyvis-gnn
```

## Usage

```bash
# Single training run
python GNN_Main.py -o generate_train_test_plot fly_N9_62_0

# LLM-guided parallel exploration
python GNN_LLM_parallel_flyvis.py -o train_test_plot_Claude_cluster fly_N9_62_0 iterations=144 --resume
```

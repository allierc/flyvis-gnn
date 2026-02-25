"""Quick debug script to reproduce the data_plot TextIOWrapper error from GNN_LLM."""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flyvis_gnn.config import NeuralGraphConfig
from GNN_PlotFigure import data_plot

# Usage: python debug_data_plot.py <slot_config_name>
# Example: python debug_data_plot.py flyvis_noise_005_Claude_00

config_name = sys.argv[1] if len(sys.argv) > 1 else 'flyvis_noise_005_Claude_00'

# Mimic GNN_LLM.py path construction
pre_folder = 'fly/'
config_file = f"config/{pre_folder}{config_name}.yaml"

print(f"Loading config from: {config_file}")
config = NeuralGraphConfig.from_yaml(config_file)
config.dataset = pre_folder + config.dataset
config.config_file = pre_folder + config_name
config.simulation.noise_model_level = 0.0

device = 'cuda:0'

# Test 1: without log_file (like GNN_Main.py) — should work
print("\n=== Test 1: data_plot WITHOUT log_file ===")
try:
    data_plot(
        config=config,
        config_file=pre_folder + config_name,
        epoch_list=['best'],
        style='color',
        extended='plots',
        device=device,
        log_file=None,
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()

# Test 2: with log_file (like GNN_LLM.py) — should reproduce the error
print("\n=== Test 2: data_plot WITH log_file ===")
log_file = open('/tmp/debug_analysis.log', 'w')
try:
    data_plot(
        config=config,
        config_file=pre_folder + config_name,
        epoch_list=['best'],
        style='color',
        extended='plots',
        device=device,
        log_file=log_file,
    )
    print("OK")
except Exception as e:
    print(f"FAILED: {e}")
    traceback.print_exc()
finally:
    log_file.close()

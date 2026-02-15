import numpy as np
import torch
import torch.nn as nn
from flyvis_gnn.neuron_state import NeuronState


class PDE_N9(nn.Module):
    """Ground-truth PDE for flyvis neural signal dynamics.

    Computes dv/dt = (-v + msg + e + v_rest [+ s*tanh(v)]) / tau
    where msg = sum_j w_j * f(v_j) over incoming edges.

    Uses explicit scatter_add for message passing (no PyG dependency).
    """

    def __init__(self, aggr_type="add", p=[], params=[], f=torch.nn.functional.relu, model_type=None, n_neuron_types=None, device=None):
        super().__init__()

        self.p = p
        self.f = f
        self.model_type = model_type
        self.device = device

        for key in self.p:
            self.p[key] = self.p[key].to(device)

        if 'multiple_ReLU' in model_type:
            if n_neuron_types is None:
                raise ValueError("n_neuron_types must be provided for multiple_ReLU model type")
            if params[0][0]>0:
                self.params = torch.tensor(params[0], dtype=torch.float32, device=device).expand((n_neuron_types, 1))
            else:
                self.params = torch.abs(1 + 0.5 * torch.randn((n_neuron_types, 1), dtype=torch.float32, device=device))
        else:
            self.params = torch.tensor(params, dtype=torch.float32, device=device).squeeze()

    def _compute_messages(self, v, particle_type, edge_index):
        """Compute per-edge messages and aggregate via scatter_add.

        args:
            v: (N, 1) voltage
            particle_type: (N, 1) long — neuron type indices
            edge_index: (2, E) source/destination indices

        returns:
            msg: (N, 1) aggregated messages per node
        """
        src, dst = edge_index

        v_src = v[src]
        particle_type_src = particle_type[src]

        if 'multiple_ReLU' in self.model_type:
            edge_msg = self.p["w"][:, None] * self.f(v_src) * self.params[particle_type_src.squeeze()]
        elif 'NULL' in self.model_type:
            edge_msg = 0 * self.f(v_src)
        else:
            edge_msg = self.p["w"][:, None] * self.f(v_src)

        msg = torch.zeros(v.shape[0], edge_msg.shape[1], device=self.device, dtype=v.dtype)
        msg.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_msg), edge_msg)

        return msg

    def forward(self, state: NeuronState, edge_index: torch.Tensor, has_field=False, data_id=[]):
        """Compute dv/dt from neuron state and connectivity.

        args:
            state: NeuronState with voltage, stimulus, neuron_type fields
            edge_index: (2, E) tensor of (src, dst) edge indices

        returns:
            dv: (N, 1) voltage derivative
        """
        v = state.voltage.unsqueeze(-1)
        v_rest = self.p["V_i_rest"][:, None]
        e = state.stimulus.unsqueeze(-1)
        particle_type = state.neuron_type.unsqueeze(-1).long()

        msg = self._compute_messages(v, particle_type, edge_index)
        tau = self.p["tau_i"][:, None]

        if 'tanh' in self.model_type:
            s = self.params
            dv = (-v + msg + e + v_rest + s * torch.tanh(v)) / tau
        else:
            dv = (-v + msg + e + v_rest) / tau

        return dv

    def func(self, u, type, function):
        if function == 'phi':
            if 'multiple_ReLU' in self.model_type:
                return self.f(u) * self.params[type]
            else:
                return self.f(u)
        elif function == 'update':
            v_rest = self.p["V_i_rest"][type]
            tau = self.p["tau_i"][type]
            if 'tanh' in self.model_type:
                s = self.params
                return (-u + v_rest + s * torch.tanh(u)) / tau
            else:
                return (-u + v_rest) / tau

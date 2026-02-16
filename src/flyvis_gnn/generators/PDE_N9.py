"""Backward compatibility — class moved to flyvis_gnn.generators.flyvis_ode."""
from flyvis_gnn.generators.flyvis_ode import (  # noqa: F401
    FlyVisODE as PDE_N9,
    FlyVisODE,
    group_by_direction_and_function,
    get_photoreceptor_positions_from_net,
)

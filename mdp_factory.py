from mdp import WarehouseMDP
import numpy as np


def create_mdp(grid, start_pos, packages,storages):
    return WarehouseMDP(grid, start_pos, packages, storages)

def initialize_mdp(rows, cols, start_pos, packages, storages):
    grid = np.zeros((rows, cols))
    mdp = WarehouseMDP(
        grid = grid,
        start_pos = start_pos,
        packages = packages,
        storages = storages
    )
    return mdp


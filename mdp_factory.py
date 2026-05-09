from mdp import WarehouseMDP


def create_mdp(grid, start_pos, packages,storages):
    return WarehouseMDP(grid, start_pos, packages, storages)

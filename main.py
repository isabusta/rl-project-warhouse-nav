import time

import numpy as np
from mdp import WarehouseMDP, ACTION_NAMES
from algorithms import sarsa, backwards_induction
from visualization import plot_rewards, plot_optimal_policy, plot_values, plot_policy, plot_value_heatmap

if __name__ == "__main__":
    # ── Warehouse setup
    start_pos = (0, 0)
    packages = {0: (0, 4), 1: (1, 4), 2: (2, 5)}
    storages = {0: (3, 0), 1: (3, 2), 2: (5, 5)}

    mdp = WarehouseMDP(
        grid=[
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ],
        start_pos=start_pos,
        packages=packages,
        storages=storages,
    )


    print("=" * 50)
    print(f"Warehouse MDP")
    print(f"  States  : {mdp.n_states}")
    print(f"  Actions : {mdp.n_actions}")
    print(f"  P shape : {mdp.P.shape}")
    print(f"  R shape : {mdp.R.shape}")
    print("=" * 50)

    # Show values and policy at t=0 for the starting state
    start_state = mdp.reset()
    s0 = mdp.state_index[start_state]


    V, policy = backwards_induction(mdp, 10)
    plot_values(V, False)
    plot_policy(policy)
    plot_value_heatmap(V)



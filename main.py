import time

import numpy as np
from mdp import WarehouseMDP, ACTION_NAMES
from algorithms import backwards_induction, value_iteration, policy_iteration, q_learning, sarsa
from mdp_factory import initialize_mdp
from visualization import animate_agent, visualize_learning, plot_rewards, plot_steps

if __name__ == "__main__":
    # ── Warehouse setup
    start_pos = (0, 0)
    packages = {0: (0, 1), 1: (3, 3), 2: (4, 5)}
    storages = {0: (3, 0), 1: (3, 2), 2: (5, 5)}

    print(packages.keys())
    mdp = initialize_mdp(6, 6, start_pos, packages=packages, storages=storages)
    mdp.add_obstacles([2, 2, 3, 3], [0, 3, 5, 5])

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
    print(f"\n--- SARSA-Learning ---")
    Q, policies, rewards = sarsa(mdp, mdp.gamma)

    plot_rewards("SARSA", policies, rewards)


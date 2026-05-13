import time

import numpy as np
from mdp import WarehouseMDP, ACTION_NAMES
from algorithms import backwards_induction, value_iteration, policy_iteration, q_learning, sarsa
from mdp_factory import initialize_mdp
from visualization import animate_agent, visualize_learning, plot_rewards, plot_steps, plot_optimal_policy, \
    plot_policy_changes, plot_v_value_convergence, plot_mean_v_values, plot_policy_rewards

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

    Q, rewards, policies, _ = sarsa(mdp)

    last_idx = max(policies.keys())
    policy = policies[last_idx]

    plot_rewards("SARSA", rewards, plot_in_streamlit=False)
    plot_optimal_policy("SARSA", mdp, policy)



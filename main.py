import time

import numpy as np
from mdp import WarehouseMDP, ACTION_NAMES
from algorithms import backwards_induction, value_iteration, policy_iteration, q_learning
from visualization import animate_agent, visualize_learning

# ── Warehouse setup
mdp = WarehouseMDP(
    grid=[
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
    ],
    start_pos=(0, 0),
    packages={0: (0, 1), 1: (3, 3)},
    storages={0:  (3, 0), 1: (3, 2)},
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


# Q-Learning
print(f"\n--- Q-Learning Iteration ---")
Q, reward, policies = q_learning(mdp)
policy_ql = np.argmax(Q, axis=1)
print(f"Best action at start: {ACTION_NAMES[policy_ql[s0]]}")

start_state = mdp.reset()
visualize_learning(mdp, policies)


import numpy as np
from mdp import WarehouseMDP, ACTION_NAMES
from algorithms import backwards_induction, value_iteration, policy_iteration

# ── Warehouse setup
mdp = WarehouseMDP(
    grid=[
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
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

# Backward induction
T = 30
print(f"\n--- Backward Induction (T={T}) ---")
V_bi, policy_bi = backwards_induction(mdp, T)

# Show values and policy at t=0 for the starting state
start_state = mdp.reset()
s0 = mdp.state_index[start_state]
print(f"V[t=0, start] = {V_bi[0, s0]:.2f}")
print(f"Best action at start (t=0): {ACTION_NAMES[policy_bi[0, s0]]}")

# Value iteration
print(f"\n--- Value Iteration ---")
V_vi, policy_vi = value_iteration(mdp)
print(f"V[start]      = {V_vi[s0]:.2f}")
print(f"Best action at start: {ACTION_NAMES[policy_vi[s0]]}")

# Policy iteration
print(f"\n--- Policy Iteration ---")
V_pi, policy_pi = policy_iteration(mdp)
print(f"V[start]      = {V_pi[s0]:.2f}")
print(f"Best action at start: {ACTION_NAMES[policy_pi[s0]]}")

#  Compare
print(f"\n--- Comparison ---")
print(f"VI vs PI max|V diff| : {np.max(np.abs(V_vi - V_pi)):.6f}")
print(f"Policies agree       : {np.all(policy_vi == policy_pi)}")

#  Follow policy_vi for one episode 
print(f"\n--- Episode following Value Iteration policy ---")
state = mdp.reset()
total = 0
print(f'{"Step":<5} {"Action":<10} {"Reward":>7}')
print("-" * 25)
for step in range(50):
    s    = mdp.state_index[state]
    a    = policy_vi[s]
    state, reward, done = mdp.step(state, a)
    total += reward
    print(f"{step+1:<5} {ACTION_NAMES[a]:<10} {reward:>+7.0f}")
    if done:
        break
print("-" * 25)
print(f"Total reward : {total:.0f}  |  Steps : {step+1}")

import numpy as np


def backwards_induction(mdp, T):
    """
    Finite-horizon backward induction

    V_T(s) = 0  for all s  (terminal boundary)
    V_t(s) = max_a [ R(s,a) + gamma * sum_{s'} P(s'|s,a) * V_{t+1}(s') ]

    Returns

    V      : np.ndarray  shape (T, n_states)
    policy : np.ndarray  shape (T, n_states)  — action index per state per step
    """
    V      = np.zeros((T, mdp.n_states))
    policy = np.zeros((T, mdp.n_states), dtype=int)

    V_next = np.zeros(mdp.n_states)   # V_T = 0

    for t in range(T - 1, -1, -1):
        # Q[s, a] = R[s, a] + gamma * sum_{s'} P[s, a, s'] * V_next[s']
        Q = mdp.R + mdp.gamma * (mdp.P @ V_next)   # shape (n_states, n_actions)
        V[t]      = Q.max(axis=1)
        policy[t] = Q.argmax(axis=1)
        V_next    = V[t]

    return V, policy


def value_iteration(mdp, theta=1e-4, max_iter=1000):
    """
    Infinite-horizon value iteration (visualization notebook style).

    Converges when max|V_new - V| < theta.

    Returns

    V      : np.ndarray  shape (n_states,)
    policy : np.ndarray  shape (n_states,)
    """
    V = np.zeros(mdp.n_states)

    for it in range(1, max_iter + 1):
        Q     = mdp.R + mdp.gamma * (mdp.P @ V)   # (n_states, n_actions)
        V_new = Q.max(axis=1)
        delta = np.max(np.abs(V_new - V))
        V     = V_new
        if delta < theta:
            print(f"Value iteration converged in {it} iterations  (Δ={delta:.2e})")
            break

    policy = (mdp.R + mdp.gamma * (mdp.P @ V)).argmax(axis=1)
    return V, policy


def policy_evaluation(mdp, policy, theta=1e-4, max_iter=1000):
    """Full policy evaluation until convergence."""
    V = np.zeros(mdp.n_states)
    s_idx = np.arange(mdp.n_states)

    for _ in range(max_iter):
        V_new = mdp.R[s_idx, policy] + mdp.gamma * (mdp.P[s_idx, policy] * V).sum(axis=1)
        if np.max(np.abs(V_new - V)) < theta:
            break
        V = V_new

    return V


def policy_iteration(mdp, theta=1e-4, max_iter=100):
    """
    Infinite-horizon policy iteration (visualization notebook style).

    Returns
    -------
    V      : np.ndarray  shape (n_states,)
    policy : np.ndarray  shape (n_states,)
    """
    policy = np.zeros(mdp.n_states, dtype=int)

    for it in range(1, max_iter + 1):
        V          = policy_evaluation(mdp, policy, theta)
        Q          = mdp.R + mdp.gamma * (mdp.P @ V)
        new_policy = Q.argmax(axis=1)

        if np.all(new_policy == policy):
            print(f"Policy iteration converged in {it} iterations")
            break
        policy = new_policy

    return V, policy

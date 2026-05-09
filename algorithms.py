import numpy as np
from numpy import random


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

        policy = (mdp.R + mdp.gamma * (mdp.P @ V)).argmax(axis=1)

        if delta < theta:
            print(f"Value iteration converged in {it} iterations  (Δ={delta:.2e})")
            break

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


def q_learning(
        mdp,
        n_episodes=5000,
        alpha=0.1,  # Lernrate
        epsilon=1.0,  # Exploration-Rate
        epsilon_decay=0.995,
        min_epsilon=0.01
):
    # Initialisierung der Q-Tabelle mit Nullen
    # Größe: [Anzahl Zustände x Anzahl Aktionen]
    Q = np.zeros((mdp.n_states, mdp.n_actions))

    rewards_per_episode = []

    for episode in range(n_episodes):
        state = mdp.reset()
        state_idx = mdp.state_index[state]
        total_reward = 0
        done = False

        while not done:
            if random.random() < epsilon:
                action = random.randint(0, mdp.n_actions - 1)
            else:
                action = np.argmax(Q[state_idx])

            next_state, reward, done = mdp.step(state, action)
            next_state_idx = mdp.state_index[next_state]

            best_next_action = np.argmax(Q[next_state_idx])
            td_target = reward + mdp.gamma * Q[next_state_idx, best_next_action]
            td_error = td_target - Q[state_idx, action]
            Q[state_idx, action] += alpha * td_error

            state = next_state
            state_idx = next_state_idx
            total_reward += reward

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.3f}")

    return Q, rewards_per_episode
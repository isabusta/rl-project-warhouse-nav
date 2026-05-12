import numpy as np
from numpy import random

from mdp import WarehouseMDP


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

def q_learning(mdp: WarehouseMDP,
               n_episodes=100,
               max_steps=100,
               alpha=0.1,
               epsilon=1.0,
               epsilon_decay=0.995,
               min_epsilon=0.01):

    Q = np.zeros((mdp.n_states,
                  mdp.n_actions))

    rewards_per_episode = []
    policies = {}

    for episode in range(n_episodes):

        state = mdp.reset()
        state_idx = mdp.state_index[state]
        total_reward = 0

        for step in range(max_steps):

            # get valid actions for current state
            mask = mdp.action_masks(state)
            valid_actions = np.where(mask == 1)[0]

            # epsilon greedy — only explore/exploit among valid actions
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                # set invalid actions to -inf so argmax never picks them
                q_masked = np.where(mask, Q[state_idx], -np.inf)
                action = np.argmax(q_masked)

            next_state, reward, done = mdp.step(state, action)
            next_state_idx = mdp.state_index[next_state]

            # Q update
            if done:
                td_target = reward
            else:
                # bootstrap only from valid actions in next state
                next_mask = mdp.action_masks(next_state)
                next_valid = np.where(next_mask == 1)[0]
                best_next = next_valid[np.argmax(Q[next_state_idx, next_valid])]
                td_target = reward + mdp.gamma * Q[next_state_idx, best_next]

            Q[state_idx, action] += alpha * (td_target - Q[state_idx, action])

            state = next_state
            state_idx = next_state_idx
            total_reward += reward

            if done:
                break

        policies[episode] = np.argmax(Q, axis=1)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

    return Q, rewards_per_episode, policies

def sarsa(mdp: WarehouseMDP, gamma, epsilon = 0.1, alpha = 0.1, episodes=100, max_iter=100):

    Q = np.zeros((mdp.n_states, mdp.n_actions))

    rewards_per_episode = []
    policies = {}

    for episode in range(episodes):
        state = mdp.reset()
        state_idx = mdp.state_index[state]

        total_reward = 0

        action = epsilon_greedy(epsilon, mdp, Q, state_idx)

        for i in range(max_iter):

            next_state, reward, done = mdp.step(state, action)
            next_state_idx = mdp.state_index[next_state]

            next_action = epsilon_greedy(epsilon, mdp, Q, next_state_idx)

            total_reward += reward

            if done:
                Q[state_idx, action] += alpha * (reward - Q[state_idx, action])
                break

            Q[state_idx, action] = Q[state_idx, action] + alpha * (reward + gamma * Q[next_state_idx, next_action] - Q[state_idx, action])

            state = next_state
            state_idx = next_state_idx
            action = next_action

        policies[episode] = np.argmax(Q, axis=1)
        rewards_per_episode.append(total_reward)

    return Q, policies, rewards_per_episode

def epsilon_greedy(epsilon, mdp, Q, state):
    # epsilon greedy
    if random.random() < epsilon:
        action = random.randint(0, mdp.n_actions - 1)
    else:
        action = np.argmax(Q[state])
    return action


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


def compare_policies(policy1, policy2):
    return policy1 == policy2


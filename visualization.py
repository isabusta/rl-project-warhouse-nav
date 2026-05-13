import matplotlib.animation as animation
import numpy as np

from mdp import WarehouseMDP, ACTION_NAMES
import matplotlib.pyplot as plt
import streamlit as st


def animate_agent(mdp: WarehouseMDP, policy, added_obstacle = None, plot_in_streamlit=False, animation_speed=200):
    start_state = mdp.reset()
    path_states, _, _ = follow_policy(mdp, policy, start_state)

    # 2. Plot Setup
    fig, ax = plt.subplots(figsize=(mdp.nrows + 2, mdp.ncols + 2))

    def update(frame_idx):
        ax.clear()
        current_state = path_states[frame_idx]
        (curr_row, curr_col), carrying, delivered = current_state

        if added_obstacle is not None:
            col, row = added_obstacle
            obstacle = plt.Circle((col, row), 0.2, color='black', ec='black', lw=2, zorder=10)
            ax.add_patch(obstacle)

        obstacles = np.argwhere(mdp.grid == 1)
        for row, col in obstacles:
            obstacle = plt.Circle((col, row), 0.2, color='black', ec='black', lw=2, zorder=10)
            ax.add_patch(obstacle)

        ax.set_aspect('equal')

        ax.set_xticks(np.arange(-0.5, mdp.ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, mdp.nrows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        for pid, loc in mdp.packages.items():
            if not delivered[pid] and carrying != pid:
                ax.plot(loc[1], loc[0], 'bs', markersize=15, label=f"Paket {pid}")
                ax.text(loc[1], loc[0], f"P{pid}", color='white', ha='center', va='center', fontweight='bold')

        # Storages zeichnen
        for pid, loc in mdp.storages.items():
            color = 'green' if delivered[pid] else 'red'
            ax.plot(loc[1], loc[0], 'o', color=color, markersize=20, alpha=0.6)
            ax.text(loc[1], loc[0], f"S{pid}", ha='center', va='center', fontweight='bold')

        # Agent zeichnen (Farbe ändert sich, wenn er trägt)
        agent_color = 'gold' if carrying is not None else 'royalblue'
        agent_circle = plt.Circle((curr_col, curr_row), 0.15, color=agent_color, ec='black', lw=2, zorder=10)
        ax.add_patch(agent_circle)

        if carrying is not None:
            ax.text(curr_col, curr_row, f"P{carrying}", ha='center', va='center', fontweight='bold')
            ax.set_title(f"Schritt: {frame_idx} | Trägt: Package P{carrying} | Geliefert: {list(delivered)}")
        else:
            ax.set_title(f"Schritt: {frame_idx} | No package carrying | Geliefert: {list(delivered)}")

        ax.set_xlim(-0.5, mdp.ncols - 0.5)
        ax.set_ylim(mdp.nrows - 0.5, -0.5)

        if frame_idx == len(path_states) - 1:
            plt.pause(0.5)
            plt.close(fig)

    # Animation erstellen
    ani = animation.FuncAnimation(
        fig, update, frames=len(path_states), interval=animation_speed, repeat=False
    )
    ax.grid(True)


    plt.grid(True)
    plt.show()
    plt.close(fig)

    return ani

def visualize_learning(mdp: WarehouseMDP, policies, added_obstacles) -> None:
    for episode, policy in policies.items():
        print(f"\n--- Episode following policy ---")
        print("-" * 25)

        start_state = mdp.reset()
        if added_obstacles[episode] is not None:
            obstacle = added_obstacles[episode]
            animate_agent(mdp, policy, start_state, obstacle)


def plot_moving_average_rewards(rewards, window=50, plot_in_streamlit=False):
    rewards = np.array(rewards)

    moving_avg = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    fig = plt.figure(figsize=(8,5))
    plt.plot(rewards, alpha=0.3, label="Raw rewards")
    plt.plot(
        range(window-1, len(rewards)),
        moving_avg,
        label=f"Moving average ({window})"
    )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Learning Curve")
    plt.legend()
    plt.tight_layout()
    if plot_in_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

# plot a line how the reward changes over the episodes
def plot_rewards(algorithm, rewards, plot_in_streamlit=False):

    episodes = len(rewards)

    fig, ax = plt.subplots()
    ax.plot(np.arange(episodes), rewards)

    ax.set_title(f"{algorithm} | Total Rewards per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")

    if plot_in_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

def plot_steps(algorithm: str, mdp: WarehouseMDP, policies, plot_in_streamlit=False):

    steps = []
    # max number of steps
    max_steps = mdp.nrows * mdp.ncols
    start_state = mdp.reset()

    # compute number of steps to reach goal
    for policy in policies.values():
        _, n_steps, _ = follow_policy(mdp, policy, start_state=start_state)
        steps.append(n_steps)

    fig, ax = plt.subplots()

    episodes = np.arange(len(policies))

    ax.plot(episodes, steps)

    ax.set_title(f"{algorithm} | Steps per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps until goal")

    ax.set_yticks(np.arange(max_steps, step=2))

    ax.grid(True)

    if plot_in_streamlit:
        st.pyplot(fig)
    else:
        plt.show()
    plt.close(fig)

def plot_policy_changes(policies: dict, plot_in_streamlit=False):
    """
    policies: {episode: policy_array}
    """

    episodes = sorted(policies.keys())
    change_rates = []

    prev_policy = None

    for ep in episodes:
        policy = policies[ep]

        if prev_policy is None:
            change_rates.append(0)
        else:
            changes = np.sum(policy != prev_policy)

            # normiert auf Anzahl States
            change_rate = changes / len(policy)
            change_rates.append(change_rate)

        prev_policy = policy

    fig = plt.figure(figsize=(8, 4))
    plt.plot(episodes, change_rates)
    plt.xlabel("Episode")
    plt.ylabel("Policy Change Rate")
    plt.title("Policy Stability over Time")
    plt.ylim(0, np.percentile(change_rates, 95) + 0.05)
    plt.grid(True)
    if plot_in_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

def plot_success_rate(mdp, policies, max_steps=None, plot_in_streamlit=False):
        successes = 0
        success_rates = []
        episodes = []

        for i, (episode, policy) in enumerate(policies.items(), start=1):

            _, steps, done = follow_policy(
                mdp,
                policy,
                start_state=mdp.reset()
            )

            if done and (max_steps is None or steps <= max_steps):
                successes += 1

            success_rate = successes / i

            episodes.append(episode)
            success_rates.append(success_rate)

        # Plot
        fig = plt.figure(figsize=(5, 5))
        plt.plot(episodes, success_rates)
        plt.ylabel("Success Rate")
        plt.xlabel("Episode")
        plt.title("Running Success Rate")
        plt.ylim(0, 1)
        plt.tight_layout()
        if plot_in_streamlit:
            st.pyplot(fig)
        else:
            plt.show()

# Helper function to compute number of steps in each policy
def follow_policy(mdp: WarehouseMDP, policy, start_state):

    path_states = [start_state]
    state = start_state
    steps = mdp.nrows * mdp.ncols
    done = False
    total_reward = 0

    for step in range(steps):
        s = mdp.state_index[state]
        a = policy[s]
        state, reward, done = mdp.step(state, a)
        total_reward += reward
        path_states.append(state)
        if done:
            return path_states, step + 1, done
    
    return path_states, steps, done

def plot_optimal_policy(algorithm: str, mdp: WarehouseMDP, policy, plot_in_streamlit=False):

    start_state = mdp.reset()
    path_states, _, _ = follow_policy(mdp, policy, start_state)

    # 2. Plot Setup
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(f"Optimal Policy for {algorithm}")


    obstacles = np.argwhere(mdp.grid == 1)
    for row, col in obstacles:
        circle = plt.Circle((col, row), 0.2, color='black', ec='black', lw=2, zorder=10)
        ax.add_patch(circle)

    ax.set_aspect('equal')

    ax.set_xticks(np.arange(-0.5, mdp.ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mdp.nrows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    for pid, loc in mdp.packages.items():
        ax.plot(loc[1], loc[0], 'bs', markersize=15, label=f"Paket {pid}")
        ax.text(loc[1], loc[0], f"P{pid}", color='white', ha='center', va='center', fontweight='bold')

    # Storages zeichnen
    for pid, loc in mdp.storages.items():
        color = 'green'
        ax.plot(loc[1], loc[0], 'o', color=color, markersize=20, alpha=0.6)
        ax.text(loc[1], loc[0], f"S{pid}", ha='center', va='center', fontweight='bold')

    xs = []
    ys = []

    for s in path_states:
        (row, col), carrying, delivered = s
        xs.append(col)
        ys.append(row)
        ax.plot(xs, ys, marker="o", color="red", linewidth=2)

    print(xs)
    ax.set_xlim(-0.5, mdp.ncols - 0.5)
    ax.set_ylim(mdp.nrows - 0.5, -0.5)

    ax.grid(True)

    if plot_in_streamlit:
        st.pyplot(fig)
    else:
        plt.show()



def plot_v_value_convergence(V_history, plot_in_streamlit=False):

    iterations = sorted(V_history.keys())
    print(iterations)
    diffs = []

    prev_V = None

    for i in iterations:
        V = np.array(V_history[i])

        if prev_V is None:
            diffs.append(0)
        else:
            diffs.append(np.max(np.abs(V - prev_V)))

        prev_V = V

    fig, ax = plt.subplots()

    ax.plot(iterations, diffs)
    ax.set_title("Policy Iteration | V Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("max |ΔV|")
    ax.grid(True)

    if plot_in_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

    plt.close(fig)


def plot_mean_v_values(V_values, plot_in_streamlit=False):


    iterations = sorted(V_values.keys())
    means = []

    for it in iterations:
        means.append(np.mean(V_values[it]))

    fig, ax = plt.subplots()

    ax.plot(iterations, means, marker="o")

    ax.set_title("Policy Iteration | Mean V Value per Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("mean V(s)")
    ax.grid(True)

    if plot_in_streamlit:
        st.pyplot(fig)
    else:
        plt.show()

    plt.close(fig)


def plot_policy_rewards(algorithm, mdp, policies):
    rewards = []

    max_steps = mdp.nrows * mdp.ncols

    for episode, policy in policies.items():

        state = mdp.reset()
        total_reward = 0

        for step in range(max_steps):

            s = mdp.state_index[state]
            a = policy[s]

            state, reward, done = mdp.step(state, a)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)

    plot_rewards(algorithm, rewards)


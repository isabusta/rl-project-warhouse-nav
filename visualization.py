import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from mdp import WarehouseMDP, ACTION_NAMES


def animate_agent(mdp: WarehouseMDP, policy, start_state):

    path_states = [start_state]
    state = start_state
    total = 0

    for step in range(30):
        s = mdp.state_index[state]
        a = policy[s]
        state, reward, done = mdp.step(state, a)
        total += reward
        print(f"{step + 1:<5} {ACTION_NAMES[a]:<10} {reward:>+7.0f}")
        path_states.append(state)
        if done:
            break

    # 2. Plot Setup
    fig, ax = plt.subplots(figsize=(mdp.nrows + 2, mdp.ncols + 2))

    def update(frame_idx):
        ax.clear()
        current_state = path_states[frame_idx]
        (row, col), carrying, delivered = current_state

        #ax.imshow(mdp.grid, cmap="gray_r", origin="upper")
        obstacles_x = np.where(mdp.grid == 1)[0]
        obstacles_y = np.where(mdp.grid == 1)[1]
        for i in range(len(obstacles_x)):
            agent_circle = plt.Circle((obstacles_x[i], obstacles_y[i]), 0.2, color='black', ec='black', lw=2, zorder=10)
            ax.add_patch(agent_circle)

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
        agent_circle = plt.Circle((col, row), 0.15, color=agent_color, ec='black', lw=2, zorder=10)
        ax.add_patch(agent_circle)

        if carrying is not None:
            ax.text(col, row, f"P{carrying}", ha='center', va='center', fontweight='bold')
            ax.set_title(f"Schritt: {frame_idx} | Trägt: Package P{carrying} | Geliefert: {list(delivered)}")
        else:
            ax.set_title(f"Schritt: {frame_idx} | No package carrying | Geliefert: {list(delivered)}")


        ax.set_title(f"Schritt: {frame_idx} | Trägt: Package P{carrying} | Geliefert: {list(delivered)}")
        ax.set_xlim(-0.5, mdp.ncols - 0.5)
        ax.set_ylim(mdp.nrows - 0.5, -0.5)

        if frame_idx == len(path_states) - 1:
            plt.pause(0.5)
            plt.close(fig)

    # Animation erstellen
    ani = animation.FuncAnimation(
        fig, update, frames=len(path_states), interval=50, repeat=False
    )
    ax.grid(True)

    plt.grid(True)
    plt.show()
    plt.close(fig)

    return total

def visualize_learning(mdp: WarehouseMDP, policies) -> None:
    rewards = {}
    for episode, policy in policies.items():
        print(f"\n--- Episode following policy ---")
        print("-" * 25)
        start_state = mdp.reset()
        rewards[episode] = animate_agent(mdp, policy, start_state)


# plot a line how the reward changes over the episodes
def plot_rewards(algorithm: str, policies):
    episodes = len(policies)
    pass

# Todo plot the number steps until goal is achieved
def plot_steps(algorithm: str, policies):
    pass


def plot_policy_changes():
    pass


def plot_success_rate():
    pass



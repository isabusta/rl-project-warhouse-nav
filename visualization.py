import matplotlib.pyplot as plt
import numpy as np

def render_value_grid(mdp, V, policy):
    rows, cols = mdp.grid.shape

    fig, ax = plt.subplots()

    # Heatmap (Value Function)
    value_grid = project_value_to_grid(mdp, V)

    im = ax.imshow(value_grid, cmap="viridis")
    plt.colorbar(im, ax=ax)

    # Walls
    for r in range(rows):
        for c in range(cols):
            if mdp.grid[r][c] == 1:
                ax.text(c, r, "X", ha="center", va="center", color="red")

    # Policy arrows
    actions = {
        0: (0, -1),  # left
        1: (0, 1),   # right
        2: (-1, 0),  # up
        3: (1, 0),   # down
    }

    for r in range(rows):
        for c in range(cols):
            a = policy[r * cols + c]
            dr, dc = actions[a]

            ax.arrow(c, r, dc*0.3, dr*0.3,
                     head_width=0.1,
                     color="white")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Value Iteration")

    return fig


def project_value_to_grid(mdp, V):
    grid = np.zeros((mdp.nrows, mdp.ncols))
    counts = np.zeros((mdp.nrows, mdp.ncols))

    for i, state in enumerate(mdp.states):
        (r, c), carrying, delivered = state

        grid[r, c] += V[i]
        counts[r, c] += 1

    return grid / np.maximum(counts, 1)
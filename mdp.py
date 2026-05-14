import numpy as np
from typing import Dict, List, Optional, Tuple

# Reward constants
STEP_COST       = -1
WALL_PENALTY    = -5
INVALID_PENALTY = -10
PICKUP_REWARD   = +10
DELIVERY_REWARD = +100

# Action indices (matching course convention)
UP      = 0
DOWN    = 1
LEFT    = 2
RIGHT   = 3
PICKUP  = 4
DELIVER = 5
N_ACTIONS = 6

ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
ACTION_NAMES  = {UP: 'up', DOWN: 'down', LEFT: 'left', RIGHT: 'right',
                 PICKUP: 'pickup', DELIVER: 'deliver'}


class WarehouseMDP:
    """
    Warehouse grid-world MDP.


    State: (robot_pos, carrying, delivered)
      robot_pos  : (row, col)
      carrying   : None or package id (int)
      delivered  : tuple[bool, ...]  — one entry per package

    Actions: 0=up  1=down  2=left  3=right  4=pickup  5=deliver
    """

    def __init__(
        self,
        grid: List[List[int]],
        start_pos: Tuple[int, int],
        packages: Dict[int, Tuple[int, int]],   # pid -> pickup (row,col)
        storages: Dict[int, Tuple[int, int]],    # pid -> storage (row,col)
        gamma: float = 0.95,
    ):
        self.grid       = np.array(grid)
        self.nrows, self.ncols = self.grid.shape
        self.start_pos  = start_pos
        self.packages   = packages
        self.storages   = storages
        self.n_packages = len(packages)
        self.gamma      = gamma
        self.n_actions  = N_ACTIONS

        self.states      = self._enumerate_states()
        self.n_states    = len(self.states)
        self.state_index = {s: i for i, s in enumerate(self.states)}

        # P[s, a, s'] and R[s, a]
        self.P, self.R = self._build_matrices()

    # State enumeration
    def _enumerate_states(self):
        passable = [
            (r, c)
            for r in range(self.nrows)
            for c in range(self.ncols)
            if self.grid[r, c] == 0
        ]
        states = []
        for pos in passable:
            for carrying in [None] + list(self.packages.keys()):
                for bits in range(2 ** self.n_packages):
                    delivered = tuple(bool((bits >> i) & 1) for i in range(self.n_packages))
                    # skip impossible states: can't carry an already-delivered package
                    if carrying is not None and delivered[carrying]:
                        continue
                    states.append((pos, carrying, delivered))
        return states

    # ── MazeEnv-style interface
    def is_terminal(self, state) -> bool:
        _, _, delivered = state
        return all(delivered)

    def reward(self, state, action: int) -> float:
        if self.is_terminal(state):
            return 0.0
        pos, carrying, delivered = state
        row, col = pos

        if action < 4:  # movement
            dr, dc = ACTION_DELTAS[action]
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.nrows and 0 <= nc < self.ncols and self.grid[nr, nc] == 0:
                return float(STEP_COST)
            return float(WALL_PENALTY)

        if action == PICKUP:
            if carrying is None:
                for pid, loc in self.packages.items():
                    if pos == loc and not delivered[pid]:
                        return float(PICKUP_REWARD + STEP_COST)
            return float(INVALID_PENALTY)

        if action == DELIVER:
            if carrying is not None and pos == self.storages[carrying]:
                return float(DELIVERY_REWARD + STEP_COST)
            return float(INVALID_PENALTY)

        return 0.0

    def transitions(self, state, action: int) -> List[Tuple[float, tuple]]:
        """Returns [(probability, next_state)]. Deterministic → one pair."""
        if self.is_terminal(state):
            return [(1.0, state)]

        pos, carrying, delivered = state
        row, col = pos
        delivered = list(delivered)

        if action < 4:  # movement
            dr, dc = ACTION_DELTAS[action]
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.nrows and 0 <= nc < self.ncols and self.grid[nr, nc] == 0:
                return [(1.0, ((nr, nc), carrying, tuple(delivered)))]
            return [(1.0, state)]

        if action == PICKUP:
            if carrying is None:
                for pid, loc in self.packages.items():
                    if pos == loc and not delivered[pid]:
                        return [(1.0, (pos, pid, tuple(delivered)))]
            return [(1.0, state)]

        if action == DELIVER:
            if carrying is not None and pos == self.storages[carrying]:
                delivered[carrying] = True
                return [(1.0, (pos, None, tuple(delivered)))]
            return [(1.0, state)]

        return [(1.0, state)]

    # DiscreteMDP Inference
    def get_transition_probability(self, state, action: int, next_state) -> float:
        return self.P[self.state_index[state], action, self.state_index[next_state]]

    def get_transition_probabilities(self, state, action: int) -> np.ndarray:
        return self.P[self.state_index[state], action]

    def get_reward(self, state, action: int) -> float:
        return self.R[self.state_index[state], action]

    def action_masks(self, state) -> np.ndarray:
        pos, carrying, delivered = state
        row, col = pos

        # start with all actions invalid
        mask = np.zeros(self.n_actions, dtype=np.int8)

        # for each movement action, compute where the robot would land
        # and check if that cell is in bounds and not a wall
        for a, (dr, dc) in ACTION_DELTAS.items():
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.nrows and 0 <= nc < self.ncols and self.grid[nr, nc] == 0:
                mask[a] = 1

        # PICKUP is valid only if: hands are empty, standing on an undelivered package
        if carrying is None:
            for pid, loc in self.packages.items():
                if pos == loc and not delivered[pid]:
                    mask[PICKUP] = 1
                    break  # found one valid package, no need to check the rest

        # DELIVER is valid only if: holding a package and standing on its storage cell
        if carrying is not None and pos == self.storages[carrying]:
            mask[DELIVER] = 1

        return mask

    #  Gymnasium-style interface
    def reset(self):
        return (self.start_pos, None, tuple([False] * self.n_packages))

    def step(self, state, action: int):
        next_state = self.transitions(state, action)[0][1]
        r = self.reward(state, action)
        done = self.is_terminal(next_state)
        return next_state, r, done

    # Matrix construction
    def _build_matrices(self):
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))
        for i, s in enumerate(self.states):
            for a in range(self.n_actions):
                R[i, a] = self.reward(s, a)
                for prob, s_next in self.transitions(s, a):
                    P[i, a, self.state_index[s_next]] += prob
        return P, R

    def add_obstacle(self, x, y):
        if self.start_pos == (x, y):
            return

        for i in self.packages.keys():
            (px, py) = self.packages[i]
            (sx, sy) = self.storages[i]
            if (x, y ) == (px, py) or (x, y) == (sx, sy):
                return

        self.grid[x][y] = 1

        self.rebuild_mdp()

    def add_obstacles(self, x_positions, y_positions):
         assert len(x_positions) == len(y_positions)

         for i in range(len(x_positions)):
             x, y = x_positions[i], y_positions[i]

             for j in self.packages.keys():
                (px, py) = self.packages[j]
                (sx, sy) = self.storages[j]

                if (x, y) == (px, py) or (x, y) == (sx, sy):
                    continue
                else:
                    self.grid[x][y] = 1

         self.rebuild_mdp()


    def rebuild_mdp(self):
        # enumerate states
        self.states = self._enumerate_states()
        self.state_index = {s: i for i, s in enumerate(self.states)}

        # P[s, a, s'] and R[s, a]
        self.P, self.R = self._build_matrices()

    def add_random_obstacle(self):

        free_cells = np.argwhere(self.grid == 0)

        x, y = free_cells[np.random.randint(len(free_cells))]

        packages = set(self.packages.values())
        storages = set(self.storages.values())

        if (x, y) in storages or (x, y) in packages:
            return None

        self.grid[x, y] = 1
        print("added grid")
        self.rebuild_mdp()
      
        return x, y




# demo
if __name__ == '__main__':
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

    print(f'States      : {mdp.n_states}')
    print(f'Actions     : {mdp.n_actions}')
    print(f'P shape     : {mdp.P.shape}')
    print(f'R shape     : {mdp.R.shape}')
    print(f'P row sums  : all ≈ 1? {np.allclose(mdp.P.sum(axis=2), 1)}')

    # example trajectory
    plan = [
        (RIGHT,   'move to P0'),
        (PICKUP,  'pick up P0'),
        (LEFT,    'col 1 -> 0'),
        (DOWN,    'row 0 -> 1'),
        (DOWN,    'row 1 -> 2'),
        (DOWN,    'row 2 -> 3'),
        (DELIVER, 'deliver P0 at D0'),
        (RIGHT,   'col 0 -> 1'),
        (RIGHT,   'col 1 -> 2'),
        (RIGHT,   'col 2 -> 3'),
        (PICKUP,  'pick up P1'),
        (LEFT,    'col 3 -> 2'),
        (DELIVER, 'deliver P1 at D1'),
    ]

    state = mdp.reset()
    total = 0
    print(f'\n{"Step":<5} {"Action":<10} {"Note":<22} {"Reward":>7}')
    print('-' * 48)
    print(f'{"0":<5} {"—":<10} {"start":<22}         {state}')
    for i, (a, note) in enumerate(plan, 1):
        state, r, done = mdp.step(state, a)
        total += r
        print(f'{i:<5} {ACTION_NAMES[a]:<10} {note:<22} {r:>+7.0f}  {state}')
        if done:
            break
    print('-' * 48)
    print(f'Total reward: {total}  |  Done: {done}')

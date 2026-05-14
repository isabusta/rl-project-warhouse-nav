import streamlit as st
from st_selectable_grid import st_selectable_grid

from algorithms import q_learning, sarsa, policy_iteration
from mdp import WarehouseMDP
from visualization import visualize_learning, plot_rewards, plot_steps, plot_success_rate, plot_policy_changes, \
    plot_moving_average_rewards, plot_optimal_policy, plot_mean_v_values, plot_v_value_convergence, plot_policy_rewards, \
    plot_mean_rewards

st.session_state.algorithm = None

if "policies" not in st.session_state:
    st.session_state.policies = {}
if "rewards" not in st.session_state:
    st.session_state.rewards = []
if "algorithm" not in st.session_state:
    st.session_state.algorithm = None
if "V_values" not in st.session_state:
    st.session_state.V_values = {}

st.set_page_config(
    page_title="Warehouse Layout Designer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    rows         = st.number_input("Grid rows",    min_value=3, max_value=20, value=4)
    cols         = st.number_input("Grid columns", min_value=3, max_value=20, value=4)
    num_packages = st.number_input("Packages",     min_value=1, max_value=5,  value=2)

    st.info("Select a cell type, then click a grid cell to place it.")

    PACKAGE_COLORS = ['#FF9800', '#F44336', '#9C27B0', '#FF5722', '#E91E63']
    STORAGE_COLORS = ['#4CAF50', '#2E7D32', '#1B5E20', '#388E3C', '#43A047']

    categories = [
        {'name': 'empty',    'color': '#f5f5f5', 'label': ''},
        {'name': 'Start',    'color': '#2196F3', 'label': 'S'},
        {'name': 'Obstacle', 'color': '#424242', 'label': '#'},
    ]
    for i in range(num_packages):
        categories.append({'name': f'Package {i}', 'color': PACKAGE_COLORS[i % 5], 'label': f'P{i}'})
        categories.append({'name': f'Storage {i}', 'color': STORAGE_COLORS[i % 5], 'label': f'D{i}'})

    cat_name_to_index = {cat['name']: i for i, cat in enumerate(categories)}
    category_options  = [cat['name'] for cat in categories if cat['name'] != 'empty']
    current_category  = st.selectbox("Place on grid", category_options)
    selected_code     = cat_name_to_index[current_category]

# ── Grid state init / reset ─────────────────────────────────────────────
if (
    'cell_status' not in st.session_state
    or len(st.session_state.cell_status) != rows
    or len(st.session_state.cell_status[0]) != cols
    or len(st.session_state.get('last_categories', [])) != len(categories)
):
    st.session_state.cell_status     = [[0] * cols for _ in range(rows)]
    st.session_state.last_categories = list(categories)


def set_unique(r, c, code):
    for rr in range(rows):
        for cc in range(cols):
            if st.session_state.cell_status[rr][cc] == code:
                st.session_state.cell_status[rr][cc] = 0
    st.session_state.cell_status[r][c] = code


# ── Main area ───────────────────────────────────────────────────────────
st.title("Warehouse Layout Designer")
st.caption("Design your warehouse layout — the MDP is built automatically below.")
st.divider()

# Algorithms

st.subheader("Reinforcement Learning / Planning")

algorithm = st.selectbox(
    "Select algorithm",
    ["Q-Learning", "SARSA", "Value Iteration", "Policy Iteration"]
)
if algorithm == "Q-Learning" or algorithm == "SARSA":
    epsilon = st.number_input("Choose epsilon fo the training", 0.0001, 0.99)
    st.session_state.epsilon = epsilon
    add_rand_obstacle = st.checkbox("Add random obstacles")
    st.session_state.add_rand_obstacle = add_rand_obstacle



if algorithm == "Policy Iteration" or algorithm == "Value Iteration":
    theta = st.number_input("Choose Theta", 0.0001, 1)
    st.session_state.theta = theta

episodes = st.number_input("Choose number of episodes", 0, 10000, step=1)
st.session_state.episodes = episodes

start_button = st.button("🚀 Start Algorithm")

if start_button and episodes > 0:

    st.info(f"Running: {algorithm}")
    mdp = st.session_state.mdp

    if "epsilon" not in st.session_state:
        epsilon = 0.1
    else:
        epsilon = st.session_state.epsilon

    if "theta" not in st.session_state:
        theta = 1e-4
    else:
        theta = st.session_state.theta

    if "add_rand_obstacle" in st.session_state:
        add_rand_obstacle = st.session_state.add_rand_obstacle
    else:
        add_rand_obstacle = False

    # ── PLACEHOLDER DISPATCH ─────────────────────────────
    if algorithm == "Q-Learning":
        st.write("Running Q-Learning...")
        Q, rewards, policies, _ = q_learning(st.session_state.mdp, n_episodes=st.session_state.episodes, epsilon=epsilon, add_rand_obstacle=add_rand_obstacle)
        st.session_state.rewards = rewards
        st.session_state.policies = policies
        st.session_state.algorithm = algorithm
        last_episode = max(policies.keys())
        policy = policies[last_episode]
        st.session_state.last_policy = policy

    elif algorithm == "SARSA":
        st.write("Running SARSA...")
        Q, rewards, policies, _ = sarsa(st.session_state.mdp, episodes=st.session_state.episodes, epsilon=epsilon, add_rand_obstacle=add_rand_obstacle)
        st.session_state.rewards = rewards
        st.session_state.policies = policies
        st.session_state.algorithm = algorithm
        last_episode = max(policies.keys())
        policy = policies[last_episode]
        st.session_state.last_policy = policy

    elif algorithm == "Value Iteration":
        st.write("Running Value Iteration...")
        V, policy, policies, V_values = policy_iteration(mdp, max_iter=st.session_state.episodes, theta=theta)
        st.session_state.policies = policies
        st.session_state.algorithm = algorithm
        st.session_state.V_values = V_values
        last_episode = max(policies.keys())
        st.session_state.last_policy = policy

    elif algorithm == "Policy Iteration":
        st.write("Running Policy Iteration...")
        V, policy, policies, V_values = policy_iteration(mdp, max_iter=st.session_state.episodes)
        st.session_state.policies = policies
        st.session_state.algorithm = algorithm
        st.session_state.V_values = V_values
        last_episode = max(policies.keys())
        st.session_state.last_policy = policy


    st.success("Algorithm finished.")

# Legend
st.markdown("**Legend:**")
non_empty   = [cat for cat in categories if cat['name'] != 'empty']
legend_cols = st.columns(len(non_empty))
for col, cat in zip(legend_cols, non_empty):
    text_color = 'white' if cat['color'] != '#f5f5f5' else '#222'
    col.markdown(
        f"<div style='background:{cat['color']};border-radius:6px;padding:6px 10px;"
        f"text-align:center;color:{text_color};border:1px solid #aaa;'>"
        f"<b>{cat['label'] or '&nbsp;'}</b><br><small>{cat['name']}</small></div>",
        unsafe_allow_html=True,
    )

st.divider()

# Grid
cells = []
for r in range(rows):
    row_cells = []
    for c in range(cols):
        cat = categories[st.session_state.cell_status[r][c]]
        row_cells.append({
            'label':      cat['label'],
            'cell_color': cat['color'],
            'tooltip':    cat['name'],
        })
    cells.append(row_cells)

selection = st_selectable_grid(cells=cells, aspect_ratio=1.0, height=500, key='colorgrid')

if selection and 'primary' in selection:
    r = selection['primary']['y']
    c = selection['primary']['x']
    current_code = st.session_state.cell_status[r][c]
    if current_code == selected_code:
        st.session_state.cell_status[r][c] = 0
    elif selected_code == cat_name_to_index['Start']:
        set_unique(r, c, selected_code)
    else:
        st.session_state.cell_status[r][c] = selected_code

st.divider()

# ── Parse layout ────────────────────────────────────────────────────────
def parse_layout(cell_status, categories, rows, cols):
    grid      = [[0] * cols for _ in range(rows)]
    start_pos = None
    packages  = {}
    storages  = {}
    for r in range(rows):
        for c in range(cols):
            name = categories[cell_status[r][c]]['name']
            if name == 'Obstacle':
                grid[r][c] = 1
            elif name == 'Start':
                start_pos = (r, c)
            elif name.startswith('Package '):
                packages[int(name.split()[1])] = (r, c)
            elif name.startswith('Storage '):
                storages[int(name.split()[1])] = (r, c)
    return grid, start_pos, packages, storages


grid, start_pos, packages, storages = parse_layout(
    st.session_state.cell_status, categories, rows, cols
)

# ── Validate & MDP summary ───────────────────────────────────────────────
st.markdown("**MDP Summary:**")


st.markdown("### 📊 Training Results")

if st.session_state.algorithm is None:
    st.info("Run an algorithm to see results.")

else:

    # ── TWO COLUMN LAYOUT ───────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.algorithm == "Policy Iteration" or st.session_state.algorithm == "Value Iteration":

            plot_mean_v_values(
                V_values,
                True)

            plot_v_value_convergence(
                st.session_state.algorithm,
                V_values,
                True)

            plot_policy_rewards(st.session_state.algorithm,
                                st.session_state.mdp,
                                st.session_state.policies)

        else:
            plot_rewards(
                st.session_state.algorithm,
                st.session_state.rewards,
                plot_in_streamlit=True
            )

            plot_moving_average_rewards(
                st.session_state.rewards,
                100,
                plot_in_streamlit=True
            )

        plot_steps(
            st.session_state.algorithm,
            st.session_state.mdp,
            st.session_state.policies,
            plot_in_streamlit=True
        )

    with col2:

        plot_success_rate(
            st.session_state.algorithm,
            st.session_state.mdp,
            st.session_state.policies,
            plot_in_streamlit=True
        )

        plot_policy_changes(
            st.session_state.policies,
            plot_in_streamlit=True
        )

        plot_optimal_policy(
            st.session_state.algorithm,
            st.session_state.mdp,
            st.session_state.last_policy,
            plot_in_streamlit=True
        )

    with col3:
        if algorithm == "Q-Learning" or "SARSA":
            plot_mean_rewards(
                st.session_state.algorithm,
                st.session_state.rewards,
                plot_in_streamlit=True
            )



errors = []
if start_pos is None:
    errors.append("No start position — place an **S** cell.")
for i in range(num_packages):
    if i not in packages:
        errors.append(f"Missing **P{i}** (Package {i} pickup location).")
    if i not in storages:
        errors.append(f"Missing **D{i}** (Storage {i} delivery location).")

if errors:
    for e in errors:
        st.warning(e)
else:
    mdp = WarehouseMDP(
        grid=grid,
        start_pos=start_pos,
        packages=packages,
        storages=storages,
    )
    st.session_state.mdp = mdp
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("States",   mdp.n_states)
    c2.metric("Actions",  mdp.n_actions)
    c3.metric("Packages", num_packages)
    c4.metric("Start",    str(start_pos))
    st.caption(
        f"P matrix: {mdp.P.shape}  |  R matrix: {mdp.R.shape}  |  "
        f"Packages: {packages}  |  Storages: {storages}"
    )


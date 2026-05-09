import streamlit as st
from st_selectable_grid import st_selectable_grid
from mdp import WarehouseMDP

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
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("States",   mdp.n_states)
    c2.metric("Actions",  mdp.n_actions)
    c3.metric("Packages", num_packages)
    c4.metric("Start",    str(start_pos))
    st.caption(
        f"P matrix: {mdp.P.shape}  |  R matrix: {mdp.R.shape}  |  "
        f"Packages: {packages}  |  Storages: {storages}"
    )
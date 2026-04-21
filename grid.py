import streamlit as st
from st_selectable_grid import st_selectable_grid
import colorsys

st.set_page_config(
    page_title="Warehouse Layout Designer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    rows = st.number_input("Grid rows", min_value=3, max_value=20, value=10)
    cols = st.number_input("Grid columns", min_value=3, max_value=20, value=10)
    num_categories = st.number_input(
        "How many storage categories/sections do you need?",
        min_value=1, max_value=12, value=5, step=1
    )
    st.info(
        "Select the category you want to assign, then click on a cell in the grid to set/reset it. Only one 'Start' cell is allowed."
    )

    # Define categories
    categories = [
        {"name": "empty", "color": "#f5f5f5", "label": ""},
        {"name": "Start", "color": "#2196F3", "label": "S"},
    ]
    for i in range(1, num_categories + 1):
        hue = i / (num_categories + 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.5, 0.95)
        hexcolor = '#' + ''.join(f'{int(255 * x):02x}' for x in rgb)
        categories.append({
            "name": f"Category {i}",
            "color": hexcolor,
            "label": str(i)
        })

    # Dropdown: show all except 'empty'
    category_options = [cat["name"] for cat in categories if cat["name"] != "empty"]
    cat_name_to_index = {cat["name"]: i for i, cat in enumerate(categories)}
    current_category = st.selectbox(
        "Assign category to grid cell",
        category_options,
        index=1 if len(category_options) > 1 else 0
    )
    selected_code = cat_name_to_index[current_category]

# --- Init/Reset grid-state ---
if (
    "cell_status" not in st.session_state
    or len(st.session_state.cell_status) != rows
    or len(st.session_state.cell_status[0]) != cols
    or len(st.session_state.get("last_categories", [])) != len(categories)
):
    st.session_state.cell_status = [
        [0 for _ in range(cols)] for _ in range(rows)
    ]
    st.session_state.last_categories = categories

def set_start(y, x):
    for r in range(rows):
        for c in range(cols):
            if st.session_state.cell_status[r][c] == 1:
                st.session_state.cell_status[r][c] = 0
    st.session_state.cell_status[y][x] = 1

# --- Main area: Header, instructions ---
st.title("Warehouse Layout Designer")
st.caption(
    "Configure your warehouse or logistics area. "
    "Choose a category from the sidebar, then click a grid cell to set or remove it. Select 'Start' to define a single starting position."
)
st.divider()

# --- Main area: Category Legend ---
st.markdown("**Category Legend:**")
legend_cols = st.columns(len(categories) - 1)  # skip 'empty'
for i, cat in enumerate(categories):
    if cat["name"] == "empty":
        continue
    with legend_cols[i-1]:
        st.markdown(
            f"<div style='background:{cat['color']};border-radius:6px;padding:6px 10px;text-align:center;color:#222;border:1px solid #aaa;'>"
            f"<b>{cat['label']}</b><br><small>{cat['name']}</small>"
            "</div>",
            unsafe_allow_html=True,
        )

st.divider()

# --- Main area: Grid ---
cells = []
for r in range(rows):
    row = []
    for c in range(cols):
        cat = categories[st.session_state.cell_status[r][c]]
        cell = {
            "label": cat["label"],
            "cell_color": cat["color"],
            "tooltip": cat["name"],
        }
        row.append(cell)
    cells.append(row)

selection = st_selectable_grid(
    cells=cells,
    aspect_ratio=1.0,
    height=500,
    key="colorgrid"
)

if selection and "primary" in selection:
    r = selection["primary"]["y"]
    c = selection["primary"]["x"]
    current_cell_code = st.session_state.cell_status[r][c]
    if selected_code == 1:
        if current_cell_code == 1:
            st.session_state.cell_status[r][c] = 0
        else:
            set_start(r, c)
    else:
        if current_cell_code == selected_code:
            st.session_state.cell_status[r][c] = 0
        else:
            st.session_state.cell_status[r][c] = selected_code

st.divider()

# --- Main area: Show Current Plan table ---
st.markdown("**Current plan (table):**")
grid_out = [
    [categories[st.session_state.cell_status[r][c]]["name"] for c in range(cols)]
    for r in range(rows)
]
st.dataframe(grid_out, hide_index=True, use_container_width=True)
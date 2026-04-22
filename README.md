# rl-project-warhouse-nav


## Package Manager 
### 1\. INSTALL UV (once)

Windows PowerShell:
```
irm https://astral.sh/uv/install.ps1 | iex
```


Mac / Linux / Git Bash:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```


Then reopen your terminal and check:
```
uv --version
```
### 2\. CLONE THE REPOSITORY

```
git clone https://github.com/isabusta/rl-project-warhouse-nav.git
```
```
cd rl-project-warhouse-nav
```

### 3\. CREATE THE ENVIRONMENT (from lock file)


```
uv sync --frozen
```

Sync or update environment (use frozen to match exact versions):
```
uv sync --frozen
```

### ADD A NEW LIBRARY

```
uv add <library-name>

uv lock

git add pyproject.toml uv.lock

git commit -m "Add <library-name> dependency"

git push
```

## Streamlit 
Run streamlit 
```
streamlit run grid.py
```

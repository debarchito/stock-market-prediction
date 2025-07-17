## 1. Setup (direnv + NixOS)

If you have [direnv](https://direnv.net) (only supported in
[NixOS](https://nixos.org) systems) setup, you can scaffold the project
dependencies using:

```sh
direnv allow
# Sync the dependencies
uv sync
```

## 2. Setup (standard)

Install [uv](https://docs.astral.sh/uv) using:

```sh
# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# or, Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Now, scaffold the project dependencies using:

```sh
# Install Python 3.13.* required for this project
uv python install 3.13
# Active the virtual environment
uv venv
# Sync the dependencies
uv sync
```

## 3. Run notebooks

You must `cd` in the [notebooks](/notebooks) directory to not break relative
links used inside the notebooks.

```sh
cd notebooks
# Now, you can use marimo to edit them:
# e.g.
marimo edit lstm.py
# This should open a link in your default browser
```

# Installation

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

## Step 1: Clone the Repository

```bash
git clone https://github.com/Criss-Wang/PowerGym.git
cd PowerGym
```

## Step 2: Create Virtual Environment

```bash
# Create a new virtual environment
python3 -m venv .venv

# Activate it
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Upgrade pip
pip install -U pip
```

## Step 3: Install the Package

Choose the installation option that fits your needs:

```bash
# Basic installation (core framework only)
pip install -e .

# With power grid domain support
pip install -e ".[powergrid]"

# With multi-agent RL support (RLlib, PettingZoo)
pip install -e ".[multi_agent]"

# Full installation (all features)
pip install -e ".[all]"

# For development (includes testing and linting tools)
pip install -e ".[dev,all]"
```

## Step 4: Verify Installation

```bash
# Test the installation
python -c "import heron; import powergrid; print('Installation successful')"

# Run tests (optional)
pytest tests/ -v
```

## Installation Options

| Option | Description |
|--------|-------------|
| `pip install -e .` | Core HERON framework only |
| `pip install -e ".[powergrid]"` | Adds PandaPower for power systems |
| `pip install -e ".[multi_agent]"` | Adds RLlib, PettingZoo for MARL |
| `pip install -e ".[all]"` | Full installation with all features |
| `pip install -e ".[dev,all]"` | Development dependencies + all features |

## Troubleshooting

### PandaPower Issues

If you encounter issues with PandaPower installation:

```bash
pip install pandapower --upgrade
```

### RLlib/Ray Issues

For RLlib compatibility issues:

```bash
pip install "ray[rllib]==2.9.0"
```

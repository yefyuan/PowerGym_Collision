# HERON Project Makefile
# Usage: make new-project NAME=my_project DOMAIN=my_domain

.PHONY: new-project help install test clean

# Default values
NAME ?= my_project
DOMAIN ?= my_domain

help:
	@echo "HERON Makefile Commands:"
	@echo "  make new-project NAME=<name> DOMAIN=<domain>  - Create a new HERON project"
	@echo "  make install                                   - Install dependencies"
	@echo "  make test                                      - Run tests"
	@echo "  make clean                                     - Clean build artifacts"
	@echo ""
	@echo "Example:"
	@echo "  make new-project NAME=traffic_sim DOMAIN=traffic"

new-project:
	@echo "Creating new HERON project: $(NAME) with domain: $(DOMAIN)"
	@mkdir -p $(NAME)/$(DOMAIN)/agents
	@mkdir -p $(NAME)/$(DOMAIN)/envs
	@mkdir -p $(NAME)/$(DOMAIN)/utils
	@mkdir -p $(NAME)/examples
	@mkdir -p $(NAME)/tests
	@touch $(NAME)/$(DOMAIN)/__init__.py
	@touch $(NAME)/$(DOMAIN)/agents/__init__.py
	@touch $(NAME)/$(DOMAIN)/envs/__init__.py
	@touch $(NAME)/$(DOMAIN)/utils/__init__.py
	@printf '%s\n' \
		'[build-system]' \
		'requires = ["setuptools>=61.0", "wheel"]' \
		'build-backend = "setuptools.build_meta"' \
		'' \
		'[project]' \
		'name = "$(DOMAIN)"' \
		'version = "0.1.0"' \
		'description = "Custom domain using HERON framework"' \
		'requires-python = ">=3.10"' \
		'dependencies = [' \
		'    "gymnasium>=1.0.0",' \
		'    "numpy>=1.21.0",' \
		'    "pandas>=1.3.0",' \
		']' \
		'' \
		'[project.optional-dependencies]' \
		'heron = [' \
		'    "heron-marl @ git+https://github.com/Criss-Wang/PowerGym.git",' \
		']' \
		'dev = [' \
		'    "pytest>=7.0.0",' \
		']' \
		'' \
		'[tool.setuptools.packages.find]' \
		'where = ["."]' \
		'include = ["$(DOMAIN)*"]' \
		> $(NAME)/pyproject.toml
	@printf '%s\n' \
		'# $(NAME)' \
		'' \
		'A custom domain project using the HERON MARL framework.' \
		'' \
		'## Setup' \
		'' \
		'```bash' \
		'cd $(NAME)' \
		'python3 -m venv .venv' \
		'source .venv/bin/activate' \
		'pip install -e ".[heron,dev]"' \
		'```' \
		> $(NAME)/README.md
	@echo ""
	@echo "Project created successfully!"
	@echo ""
	@echo "Next steps:"
	@echo "  cd $(NAME)"
	@echo "  python3 -m venv .venv"
	@echo "  source .venv/bin/activate"
	@echo "  pip install -e \".[heron,dev]\""

install:
	pip install -e ".[all]"

test:
	pytest tests/ -v

clean:
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Contributing

We welcome contributions to HERON! This guide covers the development workflow.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Criss-Wang/PowerGym.git
cd PowerGym

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[dev,all]"
```

## Running Tests

```bash
# Run HERON core tests
pytest tests/ -v

# Run PowerGrid case study tests
pytest case_studies/power/tests/ -v

# Run all tests with coverage
pytest tests/ case_studies/power/tests/ --cov=heron --cov=powergrid --cov-report=html
```

## Code Style

We use the following tools for code quality:

```bash
# Format code with black
black heron/ case_studies/ tests/

# Lint with ruff
ruff check heron/ case_studies/ tests/

# Type check with mypy
mypy heron/ case_studies/power/powergrid/
```

## Project Structure

When contributing, please follow the project structure:

```
heron/                  # Domain-agnostic framework (core changes)
case_studies/           # Domain-specific implementations
├── power/              # PowerGrid case study
└── your_domain/        # Your new case study

tests/                  # HERON core tests
docs/                   # Documentation
```

## Contribution Types

### 1. Bug Fixes

1. Create an issue describing the bug
2. Fork the repository
3. Create a branch: `git checkout -b fix/issue-number`
4. Write tests that reproduce the bug
5. Fix the bug
6. Ensure tests pass
7. Submit a pull request

### 2. New Features

1. Discuss in an issue first
2. Create a branch: `git checkout -b feature/feature-name`
3. Implement with tests
4. Update documentation
5. Submit a pull request

### 3. New Case Studies

To add a new domain case study:

1. Create directory: `case_studies/your_domain/`
2. Follow the PowerGrid structure as a template
3. Update `pyproject.toml` to include your package
4. Add documentation

```
case_studies/your_domain/
├── your_package/
│   ├── __init__.py
│   ├── agents/           # Domain-specific agents
│   ├── envs/             # Domain-specific environments
│   └── utils/            # Domain-specific utilities
├── examples/             # Example scripts
├── tests/                # Domain tests
└── README.md             # Domain documentation
```

### 4. New Protocols

1. Implement in `heron/protocols/`
2. Add tests in `tests/protocols/`
3. Document usage in `docs/source/api/heron/protocols.rst`

### 5. Message Broker Implementations

To add a new message broker (e.g., Kafka, Redis):

1. Implement `MessageBroker` interface in `heron/messaging/`
2. Add connection handling and error recovery
3. Write integration tests
4. Document configuration

## Pull Request Guidelines

1. **Clear description**: Explain what and why
2. **Tests**: Include tests for new functionality
3. **Documentation**: Update docs for user-facing changes
4. **Single purpose**: One PR per feature/fix
5. **Clean history**: Squash commits if needed

## Code Review Process

1. Automated checks run (tests, linting)
2. Maintainer review
3. Address feedback
4. Merge when approved

## Documentation

Build documentation locally:

```bash
cd docs
make html
open build/html/index.html
```

## Questions?

- **Issues**: [GitHub Issues](https://github.com/Criss-Wang/PowerGym/issues)
- **Email**: zhenlin.wang.criss@gmail.com

Thank you for contributing to HERON!

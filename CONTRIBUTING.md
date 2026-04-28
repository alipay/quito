# Contributing to QuitoBench

Thank you for your interest in contributing to QuitoBench! We welcome contributions from the community.

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/alipay/quito-10b/issues) to report bugs or request features
- Include a clear description, steps to reproduce, and expected vs actual behavior
- For bugs, include your Python version, OS, and relevant package versions

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes with clear commit messages
4. Ensure your code follows the existing code style
5. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings to all public classes and functions
- Use `black` for formatting (`line-length = 100`)
- Use `isort` for import sorting

### Adding a New Model

1. Create a new model file in `quito/models/`
2. Inherit from `BaseModel` (or `TimeSeriesModel` / `StatisticalModel`)
3. Implement the required methods (`forward`, `loss`, `_load`)
4. Add a corresponding model config class in `quito/config/model.py`
5. The model will be auto-registered via the `REGISTRY` pattern
6. Add YAML configuration files in `configs/`
7. Update documentation in `docs/` and `README.md`

### Adding a New Metric

1. Add the metric function in `quito/metrics.py`
2. Register it in `METRIC_MAPPING`
3. Update documentation

## Development Setup

```bash
git clone https://github.com/alipay/quito-10b.git
cd quito-10b
pip install -e .
pip install -r requirements.txt
```

## License

By contributing to QuitoBench, you agree that your contributions will be licensed under the MIT License.

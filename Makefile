# VintageGAN Makefile
# Automates common development tasks

.PHONY: help install test test-quick test-full quality format lint verify clean watch hooks ci train train-auto train-quick

help:
	@echo "VintageGAN Development Commands"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install package and dependencies"
	@echo "  make hooks          Install git pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run quick tests"
	@echo "  make test-full      Run all tests"
	@echo "  make verify         Verify installation"
	@echo "  make watch          Watch files and auto-test"
	@echo ""
	@echo "Code Quality:"
	@echo "  make quality        Run all quality checks"
	@echo "  make format         Format code with black"
	@echo "  make lint           Lint code with flake8"
	@echo ""
	@echo "CI/CD:"
	@echo "  make ci             Run full CI pipeline locally"
	@echo "  make all            Run everything (format + quality + test)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove generated files"
	@echo ""
	@echo "Results:"
	@echo "  make results        Generate all results for paper"
	@echo ""
	@echo "Training (Automated):"
	@echo "  make train          Interactive training setup"
	@echo "  make train-auto     Fully automated training"
	@echo "  make train-quick    Quick test (~30 min)"

install:
	@echo "Installing VintageGAN..."
	pip install -e .
	pip install pre-commit black flake8 pytest
	@echo "✅ Installation complete!"

hooks:
	@echo "Installing git hooks..."
	python autotest.py --install-hooks
	@echo "✅ Git hooks installed!"

test:
	@echo "Running quick tests..."
	python run_tests.py --quick

test-full:
	@echo "Running full test suite..."
	python run_tests.py

verify:
	@echo "Verifying installation..."
	python verify_installation.py

watch:
	@echo "Starting watch mode (Ctrl+C to stop)..."
	python autotest.py

quality:
	@echo "Running code quality checks..."
	python autotest.py --once --quality

format:
	@echo "Formatting code with black..."
	black . --line-length 100 --exclude="venv|build|dist|\.eggs"
	@echo "✅ Code formatted!"

lint:
	@echo "Linting code with flake8..."
	flake8 . --max-line-length=100 --exclude=venv,build,dist,.eggs --ignore=E203,W503

ci:
	@echo "Running full CI pipeline locally..."
	python autotest.py --all --full
	@echo ""
	@echo "Testing package build..."
	python -m build
	@echo "✅ CI pipeline complete!"

all: format quality test-full
	@echo ""
	@echo "✅ All checks passed!"

clean:
	@echo "Cleaning generated files..."
	rm -rf build/ dist/ *.egg-info
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/
	rm -rf .pytest_cache/ .mypy_cache/
	rm -rf results/*.png results/*.json results/*.csv results/*.tex
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	@echo "✅ Cleanup complete!"

results:
	@echo "Generating results for paper..."
	python compile_notebooks.py --format html pdf
	@echo "✅ Results generated in results/ directory!"

train:
	@echo "Starting interactive training setup..."
	python setup_and_train.py

train-auto:
	@echo "Starting fully automated training..."
	python run_full_pipeline.py

train-quick:
	@echo "Starting quick test training..."
	python run_full_pipeline.py --quick-test

# Development shortcuts
dev-install: install hooks
	@echo "✅ Development environment ready!"

quick-check: format test
	@echo "✅ Quick checks passed!"

pre-commit: quality test
	@echo "✅ Pre-commit checks passed!"

pre-push: format quality test-full
	@echo "✅ Pre-push checks passed!"

.PHONY: help clean test lint format check install

# Default target
help:
	@echo "Quant System - Makefile Commands"
	@echo "================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies with uv"
	@echo ""
	@echo "Development:"
	@echo "  make format         Format code with ruff"
	@echo "  make lint           Run linter (ruff check)"
	@echo "  make check          Run format check + lint"
	@echo "  make test           Run tests with pytest"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean          Remove build artifacts and caches"
	@echo ""
	@echo "Sprint 0 (to be implemented):"
	@echo "  make data_us        Fetch US market data"
	@echo "  make data_cn        Fetch CN market data"
	@echo "  make train_us       Train Model 2 for US"
	@echo "  make train_cn       Train Model 2 for CN"
	@echo "  make opt_US         Run optimizer for US"
	@echo "  make opt_CN         Run optimizer for CN"
	@echo "  make backtest_us    Run backtest for US"
	@echo "  make backtest_cn    Run backtest for CN"
	@echo "  make report_US      Generate report for US"
	@echo "  make report_CN      Generate report for CN"
	@echo "  make e2e_us         End-to-end pipeline for US"
	@echo "  make e2e_cn         End-to-end pipeline for CN"

# Development workflow
install:
	uv sync --locked --all-extras --dev

format:
	uv run ruff format .

lint:
	uv run ruff check .

check: format lint

test:
	uv run pytest -q

test-cov:
	uv run pytest --cov=src --cov-report=term-missing --cov-report=html

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf .ruff_cache
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Sprint 0 targets (placeholders - to be implemented)
data_us:
	@echo "TODO: Implement US data ingestion"
	@echo "Command: uv run --locked python -m src.data.ingest_yf_us --config configs/data_us.yaml"

data_cn:
	@echo "TODO: Implement CN data ingestion"
	@echo "Command: uv run --locked python -m src.data.ingest_ak_cn --config configs/data_cn.yaml"

train_us:
	@echo "TODO: Implement Model 2 training for US"
	@echo "Command: uv run --locked python -m src.model2.train --config configs/model2_us.yaml"

train_cn:
	@echo "TODO: Implement Model 2 training for CN"
	@echo "Command: uv run --locked python -m src.model2.train --config configs/model2_cn.yaml"

opt_US:
	@echo "TODO: Implement optimizer for US"
	@echo "Command: uv run --locked python -m src.optimizer.solve_qp --config configs/optimizer.yaml --region US"

opt_CN:
	@echo "TODO: Implement optimizer for CN"
	@echo "Command: uv run --locked python -m src.optimizer.solve_qp --config configs/optimizer.yaml --region CN"

backtest_us:
	@echo "TODO: Implement backtest for US"
	@echo "Command: uv run --locked python -m src.backtest.run_backtest --config configs/backtest_us.yaml"

backtest_cn:
	@echo "TODO: Implement backtest for CN"
	@echo "Command: uv run --locked python -m src.backtest.run_backtest --config configs/backtest_cn.yaml"

report_US:
	@echo "TODO: Implement report generation for US"
	@echo "Command: uv run --locked python -m src.backtest.report --region US --out reports/sprint0_US/report.html"

report_CN:
	@echo "TODO: Implement report generation for CN"
	@echo "Command: uv run --locked python -m src.backtest.report --region CN --out reports/sprint0_CN/report.html"

e2e_us: data_us train_us opt_US backtest_us report_US
	@echo "US end-to-end pipeline completed"

e2e_cn: data_cn train_cn opt_CN backtest_cn report_CN
	@echo "CN end-to-end pipeline completed"

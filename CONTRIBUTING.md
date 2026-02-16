
Contributing · MD
Copy

# Contributing to Futures Foundation Model

## Ways to Contribute
- **New instruments**: Crypto, forex, commodities (GC, SI, CL)
- **Feature engineering**: Novel OHLCV-derived features, order flow proxies
- **Pretraining tasks**: Additional self-supervised objectives
- **Fine-tuning recipes**: Configs for specific strategies (ORB, ICT, mean reversion)
- **Evaluation benchmarks**: Standardized regime classification benchmarks

## Getting Started
1. Fork the repository
2. `pip install -e ".[dev]"`
3. Run tests: `python tests/test_model.py`
4. Create a branch: `git checkout -b feature/your-feature`

## Code Style
- Python 3.9+, format with `black`, lint with `ruff`
- Type hints encouraged, Google-style docstrings

## Adding a New Instrument
1. Add to `INSTRUMENT_MAP` in `features.py`
2. Verify feature derivation works
3. Add test case
4. Update README

## Adding a Fine-Tuning Strategy
1. Add config to `STRATEGY_CONFIGS` in `scripts/finetune.py`
2. Create example in `examples/`
3. Document labeling methodology
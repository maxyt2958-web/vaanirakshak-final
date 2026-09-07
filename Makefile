.PHONY: dev backend lint test eval bench benchmarks clean

dev:
	npm run dev --prefix frontend

backend:
	cd vanirakshak-backend && python run.py

lint:
	npm run lint --prefix frontend

test:
	pytest -v

eval:
	cd vanirakshak-backend && python -m vanirakshak eval

bench:
	cd vanirakshak-backend && python -m vanirakshak bench

benchmarks:
	python benchmarks/run_benchmarks.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

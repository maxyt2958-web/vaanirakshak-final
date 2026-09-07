.PHONY: dev backend lint test eval clean

dev:
	npm run dev --prefix frontend

backend:
	cd vanirakshak-backend && python run.py

lint:
	npm run lint --prefix frontend

test:
	cd vanirakshak-backend && python -m pytest tests/ -v

eval:
	cd vanirakshak-backend && python -m vanirakshak eval

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

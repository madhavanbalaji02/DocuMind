.PHONY: up down api ui mcp test smoke bench demo logs clean

up:
	docker compose up -d
	@echo ""
	@echo "  Qdrant:   http://localhost:6333"
	@echo "  Postgres: localhost:5432"
	@echo "  Redis:    localhost:6379"

down:
	docker compose down

api:
	uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8888

ui:
	streamlit run src/ui/app.py --server.port 8501

mcp:
	python -m src.mcp_server.server

test:
	pytest tests/ -v --tb=short

smoke:
	python scripts/smoke_test.py

bench:
	python scripts/benchmark.py

demo:
	python scripts/load_demo_data.py
	@echo ""
	@echo "Demo data loaded. Run 'make api' and 'make ui' in separate terminals."

logs:
	docker compose logs -f

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."

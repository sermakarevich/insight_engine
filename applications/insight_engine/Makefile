SERVICE=insight-engine

build:
	docker-compose --log-level ERROR -f ../docker-compose.yaml build ${SERVICE}
run:
	docker-compose --log-level ERROR  -f ../docker-compose.yaml run --rm ${SERVICE} bash
format:
	docker-compose --log-level ERROR run --rm ${SERVICE} sh -c "ruff check --fix && ruff format /src"
serve:
	docker-compose --log-level ERROR -f ../docker-compose.yaml run -p 8081:8081  --rm ${SERVICE} uvicorn --app-dir=. app.main:app --reload --port 8081 --host 0.0.0.0
test:
	docker-compose --log-level ERROR run --rm ${SERVICE} sh -c "cd / && pytest -s -v /src -W ignore::pytest.PytestCollectionWarning && mypy /src"
regenerate-requirements:
	docker-compose --log-level ERROR -f ../docker-compose.yaml run ${SERVICE} sh -c "cd /requirements && pip-compile --resolver=backtracking -v"

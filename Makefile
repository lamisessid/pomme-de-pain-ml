# Ports dédiés (5173/8000 sont souvent déjà utilisés par d'autres projets)
API_PORT ?= 8642
WEB_PORT ?= 5190
VENV ?= .venv
PY := $(abspath $(VENV))/bin/python

.PHONY: setup preprocess train test api web lint build

setup:            ## Installe l'environnement Python et les dépendances frontend
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r backend/requirements-dev.txt -r preprocessing/requirements.txt
	cd frontend && npm ci

preprocess:       ## Nettoyage + features + split (preprocessing/data/interim et processed)
	cd preprocessing && $(PY) scripts/run_preprocessing.py

train:            ## Entraîne le LSTM -> backend/artifacts/
	cd backend && $(PY) -m scripts.train

test:             ## Tests backend
	cd backend && $(PY) -m pytest -q

api:              ## API sur http://localhost:$(API_PORT) (docs : /docs)
	cd backend && $(PY) -m uvicorn app.main:app --reload --port $(API_PORT)

web:              ## Frontend sur http://localhost:$(WEB_PORT)
	cd frontend && VITE_API_URL=http://localhost:$(API_PORT) PORT=$(WEB_PORT) npm run dev

lint:
	cd frontend && npm run lint

build:
	cd frontend && npm run build

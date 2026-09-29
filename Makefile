PY ?= python

.PHONY: setup download prepare train evaluate app test

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .
	$(PY) -m spacy download en_core_web_sm

download:
	$(PY) scripts/download_data.py

prepare:
	$(PY) scripts/prepare_data.py

train:
	$(PY) -m stylometry.models.train

evaluate:
	$(PY) scripts/run_experiments.py
	$(PY) scripts/make_report_tables.py

app:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

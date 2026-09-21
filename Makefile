.PHONY: help install data run eval test clean

help:
	@echo "install  install dependencies"
	@echo "data     download, filter, classify and embed the corpus"
	@echo "run      launch the chat interface"
	@echo "eval     run the evaluation suite"
	@echo "test     run unit tests"
	@echo "clean    remove processed data (keeps the raw download)"

install:
	pip3 install -r requirements.txt

data:
	python3 -m src.ingest
	python3 -m src.cuisine
	python3 -m src.normalise
	python3 -m src.embed

run:
	streamlit run app.py

eval:
	python3 -m eval.evaluate

test:
	python3 -m pytest tests -q

clean:
	rm -f data/processed/*

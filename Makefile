.PHONY: help install data run cli notebook test clean

help:
	@echo "install   install dependencies"
	@echo "data      rebuild the curated corpus from the raw dataset"
	@echo "run       launch the chatbot (Streamlit)"
	@echo "cli       launch the chatbot in the terminal"
	@echo "notebook  open the walkthrough notebook"
	@echo "test      run unit tests"

install:
	pip install -r requirements.txt

data:
	python -m src.build_corpus

run:
	streamlit run app.py

cli:
	python -m src.chat_cli

notebook:
	jupyter notebook notebooks/recipe_rag_workshop.ipynb

test:
	python -m pytest tests -q

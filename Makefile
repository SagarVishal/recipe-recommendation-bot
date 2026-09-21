.PHONY: help install data notebook test clean

help:
	@echo "install   install dependencies"
	@echo "data      rebuild the curated corpus from the raw dataset"
	@echo "notebook  launch the workshop notebook"
	@echo "test      run unit tests"
	@echo "clean     remove the raw download (the curated CSVs are committed)"

install:
	pip3 install -r requirements.txt

data:
	python3 -m src.build_corpus

notebook:
	jupyter notebook notebooks/recipe_rag_workshop.ipynb

test:
	python3 -m pytest tests -q

clean:
	rm -f data/raw/*.csv

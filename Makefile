prepare-examples:
	@if [ ! -d data ]; then \
		gunzip -q data.tar.gz ; \
		tar -xf data.tar ; \
		rm data.tar ; \
	else \
		echo "The directory data already exists. Nothing left to do here 🦥 "; \
	fi

download-poetry:
	@curl -sSL https://install.python-poetry.org | python3 -

poetry-install:
	poetry install --no-root

install: download-poetry 

run-train: prepare-examples
	poetry run python campaign-classification/train-model.py run

run-classify:
	poetry run python campaign-classification/classify-examples.py

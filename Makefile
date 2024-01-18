prepare-examples:
	@if [ ! -d data ]; then \
		gunzip -q data.tar.gz ; \
		tar -xf data.tar ; \
		rm data.tar ; \
	else \
		echo "The directory data already exists."; \
	fi


run-train:
	METAFLOW_PROFILE=private python train-model.py run

classify-examples:
	python classify-examples.py

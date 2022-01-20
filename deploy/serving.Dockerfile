FROM pytorch/torchserve:0.5.2-cpu

USER root

COPY models/trained_model models/trained_model
RUN cd models && mkdir model_store
COPY src/ src/

RUN torch-model-archiver \
		--model-name "bert" \
		--version 1.0 \
		--serialized-file models/trained_model/pytorch_model.bin \
		--export-path models/model_store \
		--extra-files "models/trained_model/index_to_name.json,models/trained_model/config.json" \
		--handler src/models/torchserve_handler.py

CMD ["torchserve", \
     "--start", \
     "--model-store", \
     "models/model_store", \
     "--models", \
     "bert=bert.mar"]

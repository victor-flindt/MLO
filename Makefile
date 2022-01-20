#################################################################################
# GLOBALS                                                                       #
#################################################################################
PYTHON_INTERPRETER=python3
VERSION=$(shell git rev-parse @)
GPROJ=valiant-splicer-337909
# Check projects name by: gcloud projects list

#################################################################################
# Python	                                                                    #
#################################################################################
## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Remove compiled
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Make Dataset
data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw data/processed

## Train model
train: 
	$(PYTHON_INTERPRETER) src/models/train_model.py data/processed models

## Predict
predict:
	$(PYTHON_INTERPRETER) src/models/predict_model.py models/model.pth

#################################################################################
# Deploy	                                                                    #
#################################################################################
## Docker
build.trainer: clean
	docker build -f deploy/trainer.Dockerfile -t trainer .

push: build.trainer
	docker tag trainer gcr.io/$(GPROJ)/mlops:$(VERSION)
	docker push gcr.io/$(GPROJ)/mlops:$(VERSION)

.dumps:
	@mkdir -p deploy/.dumps

clean-deploy-files:
	rm -f deploy/.dumps/*

gen-spec: clean-deploy-files .dumps
	IMG_NAME=gcr.io/$(GPROJ)/mlops:$(VERSION) envsubst '$${IMG_NAME}' < deploy/trainertemplate.yaml > deploy/.dumps/trainer.yaml

deploy-trainer: gen-spec
	kubectl apply -f deploy/.dumps/trainer.yaml
# Above assumes that gcloud cluster is your default cluster (current-context)
# Watch logs locally: watch -n 1 kubectl logs [pods name]

run-local: build.trainer
	docker run --rm -it --name model-trainer gcr.io/$(GPROJ)/mlops:$(VERSION)

interactive: build.trainer
	docker run --rm -it --entrypoint bash --name model-trainer gcr.io/$(GPROJ)/mlops:$(VERSION)

## Update remote data using dvc
update-data: 
	dvc add data/
	git add data.dvc
	git commit -m "update dvc"
	git tag -a $(shell date | tr -d "[:space:]" | sed 's/://g') -m "update dvc"
	dvc push
	git push

train-model:
	rm -f models/trained_model/pytorch_model.bin
	rm -f models/trained_model/config.json
	python src/models/train_model.py

model-archive:
	torchserve --stop
	@mkdir -p models/model_store
	rm -f models/model_store/bert.mar
	torch-model-archiver \
		--model-name "bert" \
		--version 1.0 \
		--serialized-file models/trained_model/pytorch_model.bin \
		--export-path models/model_store \
		--extra-files "models/trained_model/index_to_name.json,models/trained_model/config.json" \
		--handler src/models/torchserve_handler.py

serve-local:
	torchserve --start --model-store models/model_store --models bert=bert.mar
# curl -X POST -H "Content-Type: application/json" --data '{"data": "you are a very nice person and everyone likes you"}' http://127.0.0.1:8080/predictions/bert

build.server: clean
	docker build -f deploy/serving.Dockerfile -t server .

push.server: build.server
	docker tag server gcr.io/$(GPROJ)/mlops:$(VERSION)
	docker push gcr.io/$(GPROJ)/mlops:$(VERSION)

.dumps:
	@mkdir -p deploy/.dumps

clean-deploy-files:
	rm -f deploy/.dumps/*

gen-spec: clean-deploy-files .dumps
	IMG_NAME=gcr.io/$(GPROJ)/mlops:$(VERSION) envsubst '$${IMG_NAME}' < deploy/deploytemplate.yaml > deploy/.dumps/server.yaml

deploy-server: gen-spec
	kubectl apply -f deploy/.dumps/server.yaml
# Above assumes that gcloud cluster is your default cluster (current-context)
# Watch logs locally: watch -n 1 kubectl logs [pods name]

run-local: build.server
	docker run --rm -it --name model-server gcr.io/$(GPROJ)/mlops:$(VERSION)

interactive: build.server
	docker run --rm -it --entrypoint bash --name model-server gcr.io/$(GPROJ)/mlops:$(VERSION)
#################################################################################
# GLOBALS                                                                       #
#################################################################################
PYTHON_INTERPRETER=python3
VERSION=$(shell git rev-parse @)
GPROJ=mlo-dtu
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
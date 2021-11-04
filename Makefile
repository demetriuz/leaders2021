.PHONY: run-server prepare-data train-lfm

build-app:
	docker-compose build app

prepare-data:
	docker-compose run app python prepare_data.py

train-lfm:
	docker-compose run app python train_lfm.py

run-server:
	docker-compose up

nodocker-prepare-data:
	python ./app/prepare_data.py

nodocker-train-lfm:
	python ./app/train_lfm.py

nodocker-run-server:
	cd ./app/ && uvicorn server:app

nodocker-fill-submission:
	python ./app/fill_submission.py

nodocker-evaluate:
	python ./app/evaluate.py

nodocker-make-validation-dict:
	python ./app/make_validation_dict.py
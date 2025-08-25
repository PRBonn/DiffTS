.PHONY: test

SHELL = /bin/sh

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

GPUS ?= 0
CONFIG ?= 'config_orchard.yaml'
CHECKPOINT ?= 'None'
WEIGHTS ?= 'None'
RUN_IN_CONTAINER = docker compose run -e PLS_CHECKPOINT=$(CHECKPOINT) -e PLS_CONFIG=$(CONFIG) -e CUDA_VISIBLE_DEVICES=$(GPUS) diffts
FORMAT ?= 'lineset'
FILTERING ?= 'true'
PARAMS ?=

build:
	docker compose build diffts --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

train:
	$(RUN_IN_CONTAINER) python3 train.py --config $(CONFIG) --checkpoint $(CHECKPOINT) $(PARAMS)

test:
	$(RUN_IN_CONTAINER) python3 train.py --test --weights ${WEIGHTS} --config $(CONFIG) $(PARAMS)

shell:
	$(RUN_IN_CONTAINER) bash

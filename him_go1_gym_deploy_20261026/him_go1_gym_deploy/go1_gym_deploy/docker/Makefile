# HIM deployment uses an explicitly prepared target Python runtime. The legacy
# Dockerfile's Torch 1.9 image is not silently installed or launched here.
PYTHON ?= python3
HIM_CONFIG ?=
DEPLOY_ROOT := $(abspath ..)

.PHONY: default help require-config check inspect observe deploy autostart build legacy-build
default: help
help:
	@echo 'make check|inspect|observe|deploy HIM_CONFIG=/absolute/config.json [PYTHON=python3]'
	@echo 'check/autostart are offline; deploy explicitly enters the R2-controlled runner.'
	@echo 'Use a compatible CPU Torch runtime and requirements-himloco.txt. No training setup.py is run.'
	@echo 'legacy-build only builds the original WTW image; it is not a validated HIM runtime.'

require-config:
	@test -n "$(HIM_CONFIG)" || (echo 'HIM_CONFIG is required' >&2; exit 2)
	@test -f "$(HIM_CONFIG)" || (echo 'HIM_CONFIG file does not exist' >&2; exit 2)

check: require-config
	"$(PYTHON)" "$(DEPLOY_ROOT)/scripts/deploy_policy.py" --config "$(HIM_CONFIG)" --check

inspect: require-config
	"$(PYTHON)" "$(DEPLOY_ROOT)/scripts/deploy_policy.py" --config "$(HIM_CONFIG)" --inspect-config

observe: require-config
	"$(PYTHON)" "$(DEPLOY_ROOT)/scripts/deploy_policy.py" --config "$(HIM_CONFIG)" --observe

deploy: require-config
	"$(PYTHON)" "$(DEPLOY_ROOT)/scripts/deploy_policy.py" --config "$(HIM_CONFIG)" --run

autostart: check

build: legacy-build
legacy-build:
	@echo 'WARNING: building original WTW Torch 1.9 image; validate/replace runtime before HIM deployment.'
	docker build -t jetson-model-deployment .

help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

baseline: ## Evaluate a rule-based control baseline
	@echo "Running RBC baseline"
	@python -m agents --agent rbc --episodes 3

setup: ## first-time setup of Conda environment
	conda env create -f environment.yml

deps: ## update Conda environment
	conda env update -f environment.yml

submit-deps:
	@pip list | grep -F aicrowd-cli || @pip install -U aicrowd-cli
	@git lfs || echo 'Please install git-lfs'
	@echo "All good!"

submit: submit-deps ## Submits an agent to the competition
	@echo "Logging into AICrowd"
	aicrowd login
	@echo "Submitting $(tag) to the competition"
	./submit.sh "$(tag)"

.PHONY: setup deps baseline help submit submit-deps
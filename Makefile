PYTHON ?= python3.11

.PHONY: install demo-data train-sv train-spoof train-fusion report plots test

install:
	$(PYTHON) -m pip install -r requirements.txt

demo-data:
	$(PYTHON) scripts/prepare_data.py --config configs/default.yaml

train-sv:
	$(PYTHON) scripts/train_baseline_sv.py --config configs/default.yaml

train-spoof:
	$(PYTHON) scripts/train_baseline_spoof.py --config configs/default.yaml

train-fusion:
	$(PYTHON) scripts/train_joint_model.py --config configs/default.yaml

report:
	$(PYTHON) scripts/generate_supervisor_report.py --config configs/default.yaml

plots:
	$(PYTHON) scripts/generate_all_plots.py --config configs/default.yaml

test:
	$(PYTHON) -m pytest

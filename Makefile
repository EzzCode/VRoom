.PHONY: clean build run_train profile_nsys profile_ncu run_train_ckpt profile_nsys_ckpt profile_ncu_ckpt

# Default variables - can be easily overridden from the command line
DATASET ?= 3dovs
CONFIG ?= gstrain/config/vroom/2d/$(DATASET)/config.json
GPU ?= 0
SUBDIR ?= baseline
PROFILE_DIR ?= diff-surfel-rasterization/profiling/$(DATASET)/$(SUBDIR)

clean:
	cd ./diff-surfel-rasterization && rm -rf build *.egg-info/

build:
	cd ./diff-surfel-rasterization && CC=gcc-11 CXX=g++-11 pip install -e . 2>&1 | tee build.log

run_train:
	python -m gstrain.trainer --config $(CONFIG) --gpu $(GPU)

profile_nsys:
	@mkdir -p $(PROFILE_DIR)
	nsys profile -o $(PROFILE_DIR)/nsys_profile --delay 75 --duration 20 -t cuda,cudnn,nvtx,osrt --force-overwrite=true python -m gstrain.trainer --config $(CONFIG)  --gpu $(GPU)

profile_ncu:
	@mkdir -p $(PROFILE_DIR)
	ncu --set full -o $(PROFILE_DIR)/ncu_profile --launch-skip 204 --kernel-name regex:"renderCUDA|preprocessCUDA|duplicateWithKeysCUDA|identifyTileRanges" --launch-count 6 --kill 1 --force-overwrite python -m gstrain.trainer --config $(CONFIG)  --gpu $(GPU)

run_train_ckpt:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Error: Please specify CHECKPOINT=/path/to/checkpoint_dir"; exit 1; fi
	python -m gstrain.trainer --config $(CONFIG)  --gpu $(GPU) --start_checkpoint $(CHECKPOINT)

profile_nsys_ckpt:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Error: Please specify CHECKPOINT=/path/to/checkpoint_dir"; exit 1; fi
	@mkdir -p $(PROFILE_DIR)
	nsys profile -o $(PROFILE_DIR)/nsys_profile --delay 90 --duration 20 -t cuda,cudnn,nvtx,osrt --force-overwrite=true python -m gstrain.trainer --config $(CONFIG)  --gpu $(GPU) --start_checkpoint $(CHECKPOINT)

profile_ncu_ckpt:
	@if [ -z "$(CHECKPOINT)" ]; then echo "Error: Please specify CHECKPOINT=/path/to/checkpoint_dir"; exit 1; fi
	@mkdir -p $(PROFILE_DIR)
	ncu --set full -o $(PROFILE_DIR)/ncu_profile --launch-skip 18 --kernel-name regex:"renderCUDA|preprocessCUDA|duplicateWithKeysCUDA|identifyTileRanges" --launch-count 6 --kill 1 --force-overwrite python -m gstrain.trainer --config $(CONFIG)  --gpu $(GPU) --start_checkpoint $(CHECKPOINT)
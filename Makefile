.PHONY: clean build run_train profile_nsys profile_ncu run_train_ckpt profile_nsys_ckpt profile_ncu_ckpt

# Default variables - can be easily overridden from the command line
DATASET ?= 3dovs
CONFIG ?= gstrain/config/vroom/2d/$(DATASET)/config.json
GPU ?= 0
SUBDIR ?= latest
PROFILE_DIR ?= diff-surfel-rasterization/profiling/$(DATASET)/$(SUBDIR)

clean:
	cd ./diff-surfel-rasterization && rm -rf build *.egg-info/

build:
	cd ./diff-surfel-rasterization && CC=gcc-11 CXX=g++-11 pip install -e . 2>&1 | tee build.log

run_train:
	python -m gstrain.trainer --config $(CONFIG) --gpu $(GPU)

profile_nsys:
	@mkdir -p $(PROFILE_DIR)
	nsys profile -o $(PROFILE_DIR)/nsys_profile --delay 90 --duration 20 -t cuda,cudnn,nvtx,osrt --force-overwrite=true python -m gstrain.trainer --config $(CONFIG)  --gpu $(GPU)

profile_ncu:
	@mkdir -p $(PROFILE_DIR)
	ncu --set full -o $(PROFILE_DIR)/ncu_profile --launch-skip 210 --kernel-name regex:"render_kernel|preprocess_kernel|frustum_cull_kernel|key_gen_kernel|compute_tile_ranges_kernel" --launch-count 7 --kill 1 --force-overwrite python -m gstrain.trainer --config $(CONFIG)  --gpu $(GPU)

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
	ncu --set full -o $(PROFILE_DIR)/ncu_profile --launch-skip 21 --kernel-name regex:"render_kernel|preprocess_kernel|frustum_cull_kernel|key_gen_kernel|compute_tile_ranges_kernel" --launch-count 7 --kill 1 --force-overwrite python -m gstrain.trainer --config $(CONFIG)  --gpu $(GPU) --start_checkpoint $(CHECKPOINT)

# --- Modal API Commands ---
MODAL_URL ?= https://galalmohamed2003--vroom-2dgs-pipeline-fastapi-entrypoint.modal.run
S3_URI ?= s3://scene-recon-assets-be/images/
JOB_ID ?= $(shell cat .modal_job_id 2>/dev/null)

# Include env without exporting to shell
-include App-Backend/.env
# Strip quotes from the included variables
API_KEY := $(patsubst "%",%,$(VROOM_API_KEY))

job_start:
	@echo "Starting job from $${S3_URI}..."
	@RES=$$(curl -s -X POST "$(MODAL_URL)/api/v1/jobs/start-recon-2dgs/json" \
	  -H "Authorization: Bearer $(API_KEY)" \
	  -H "Content-Type: application/json" \
	  -d '{ "data_source": { "s3_uri": "$(S3_URI)" } }') && \
	echo "$$RES" | jq . && \
	JID=$$(echo "$$RES" | jq -r .job_id) && \
	if [ "$$JID" != "null" ] && [ -n "$$JID" ]; then \
	    echo "$$JID" > .modal_job_id; \
	    echo "Saved JOB_ID=$$JID to .modal_job_id for future commands."; \
	fi

job_status:
	@if [ -z "$(JOB_ID)" ]; then echo "Error: Please specify JOB_ID=..."; exit 1; fi
	@curl -s -X GET "$(MODAL_URL)/api/v1/jobs/$(JOB_ID)" \
	  -H "Authorization: Bearer $(API_KEY)" | jq .

job_download:
	@if [ -z "$(JOB_ID)" ]; then echo "Error: Please specify JOB_ID=..."; exit 1; fi
	@echo "Requesting download URL for $(JOB_ID)..."
	@URL=$$(curl -s -X GET "$(MODAL_URL)/api/v1/jobs/$(JOB_ID)/download/meshes/bulk" \
	  -H "Authorization: Bearer $(API_KEY)" | jq -r .url) && \
	if [ "$$URL" = "null" ] || [ -z "$$URL" ]; then \
	    echo "Failed to get download URL. Ensure the job is COMPLETED."; \
	else \
	    echo "Download URL:"; \
	    echo "$$URL"; \
	fi
.PHONY: clean build run_train

clean:
	cd ./diff-surfel-rasterization && rm -rf build diff_surfel_rasterization.egg-info/

build:
	cd ./diff-surfel-rasterization && CC=gcc-11 CXX=g++-11 pip install -e .

run_train:
	python gs-train/trainer.py --config gs-train/config/vroom/2d/3dovs/config.yaml --scene_name bed --gpu 0
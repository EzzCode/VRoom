# VRoom

If you already cloned it without `--recursive`, pull the changes with the `.gitmodules` file and run inside the repo:

```bash
git submodule update --init --recursive
```

To build the CUDA rasterizer, run:

```bash
make clean build
```

To run training on 3dovs:

```bash
make run_train_3dovs
```
# VRoom

If you already cloned it without `--recursive`, pull the changes with the `.gitmodules` file and run inside the repo:

```bash
git submodule update --init --recursive
```

To build the CUDA rasterizer, run:

```bash
CC=gcc-11 CXX=g++-11 pip install -e .
```
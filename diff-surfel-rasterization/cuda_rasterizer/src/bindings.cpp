#include "../includes/rasterizer.cuh"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rasterize_surfels_fwd", &rasterize_surfels_fwd);
    m.def("rasterize_surfels_fwd_subsequent", &rasterize_surfels_fwd_subsequent);
    m.def("rasterize_surfels_bwd_render", &rasterize_surfels_bwd_render);
    m.def("rasterize_surfels_bwd_preprocess", &rasterize_surfels_bwd_preprocess);
    m.def("frustum_cull_surfels", &frustum_cull_surfels);
}
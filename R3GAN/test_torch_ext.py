from torch.utils.cpp_extension import load_inline

source = """
#include <torch/extension.h>
torch::Tensor add_one(torch::Tensor x) {
    return x + 1;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_one", &add_one, "Add one to a tensor");
}
"""

module = load_inline(name="test_extension", cpp_sources=source, functions="add_one", verbose=True)
import torch
print(module.add_one(torch.tensor([1,2,3])))

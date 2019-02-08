// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Tensors/TensorData.hpp>

#include <utility>

namespace CubbyDNN
{
TensorData::TensorData(std::vector<float> data, TensorShape shape_)
    : dataVec(std::move(data)), shape(std::move(shape_))
{
    // Do nothing
}
}  // namespace CubbyDNN
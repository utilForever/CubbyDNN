// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Computations/LossFunctions/LossFunctionWrapper.hpp>

namespace CubbyDNN::Compute
{
std::unordered_map<std::string, std::unique_ptr<BaseLoss<float>>>
LossFunctionWrapper::m_floatLossMap;

std::unordered_map<std::string, std::unique_ptr<BaseLoss<int>>>
LossFunctionWrapper::m_integerLossMap;
}

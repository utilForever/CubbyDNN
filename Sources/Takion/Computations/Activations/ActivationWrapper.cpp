// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <unordered_map>
#include<cubbydnn/Computations/Activations/ActivationWrapper.hpp>

namespace Takion::Compute
{
std::unordered_map<std::string, std::unique_ptr<ActivationFunc<float>>>
ActivationWrapper::m_floatActivationMap;


std::unordered_map<std::string, std::unique_ptr<ActivationFunc<int>>>
ActivationWrapper::m_integerActivationMap;
}

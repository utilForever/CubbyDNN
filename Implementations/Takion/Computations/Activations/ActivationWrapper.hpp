// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_ACTIVATIONWRAPPER_HPP
#define TAKION_COMPUTE_ACTIVATIONWRAPPER_HPP

#include <Takion/Computations/Activations/ActivationWrapperDecl.hpp>
#include <unordered_map>

namespace Takion::Compute
{
template <typename T>
std::unordered_map<std::string, std::unique_ptr<ActivationFunc<T>>>
ActivationWrapper<T>::m_activationMap;
}

#endif

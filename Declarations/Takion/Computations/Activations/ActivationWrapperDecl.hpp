// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTE_ACTIVATIONWRAPPER_HPP
#define UBBYDNN_COMPUTE_ACTIVATIONWRAPPER_HPP

#include <memory>
#include <unordered_map>
#include <Takion/Computations/Activations/Activation.hpp>

namespace Takion::Compute
{
template <typename T>
class ActivationWrapper
{
public:
    static const std::unique_ptr<ActivationFunc<float>>& GetActivation(
        std::string name)
    {
        return m_activationMap.at(name);
    }

    static void Initialize()
    {
        m_activationMap["ReLU"] = std::make_unique<ReLU<float>>();
        m_activationMap["SoftMax"] = std::make_unique<SoftMax<float>>();
    }

private:
    static std::unordered_map<std::string, std::unique_ptr<ActivationFunc<T>
                              >>
    m_activationMap;
};
}
#endif

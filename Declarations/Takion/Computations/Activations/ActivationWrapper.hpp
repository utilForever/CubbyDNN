// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTE_ACTIVATIONWRAPPER_HPP
#define UBBYDNN_COMPUTE_ACTIVATIONWRAPPER_HPP

#include <memory>
#include <cubbydnn/Computations/Activations/Activation.hpp>

namespace Takion::Compute
{
class ActivationWrapper
{
public:
    static const std::unique_ptr<ActivationFunc<float>>& GetFloatActivation(
        std::string name)
    {
        return m_floatActivationMap.at(name);
    }

    static const std::unique_ptr<ActivationFunc<int>>& GetIntegerActivation(
        std::string name)
    {
        return m_integerActivationMap.at(name);
    }

    static void Initialize()
    {
        m_floatActivationMap["ReLU"] = std::make_unique<ReLU<float>>();
        m_floatActivationMap["SoftMax"] = std::make_unique<SoftMax<float>>();

        m_integerActivationMap["ReLU"] = std::make_unique<ReLU<int>>();
        m_integerActivationMap["SoftMax"] = std::make_unique<SoftMax<int>>();
    }

private:
    static std::unordered_map<std::string, std::unique_ptr<ActivationFunc<float>>>
    m_floatActivationMap;
    static std::unordered_map<std::string, std::unique_ptr<ActivationFunc<int>>>
    m_integerActivationMap;
};
}
#endif

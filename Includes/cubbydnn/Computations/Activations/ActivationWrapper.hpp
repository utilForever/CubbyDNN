// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTE_ACTIVATIONWRAPPER_HPP
#define UBBYDNN_COMPUTE_ACTIVATIONWRAPPER_HPP

#include <memory>
#include <cubbydnn/Computations/Activations/Activation.hpp>

namespace CubbyDNN::Compute
{
class ActivationWrapper
{
public:
    static const ActivationFunc<float>* GetFloatActivation(
        std::string name)
    {
        return m_floatActivationMap.at(name);
    }

    static const ActivationFunc<int>* GetIntegerActivation(
        std::string name)
    {
        return m_integerActivationMap.at(name);
    }

    static void Initialize()
    {
        m_floatActivationMap = {
            { "ReLU", new ReLU<float>() },
            { "SoftMax", new SoftMax<float>() },
        };

        m_integerActivationMap = {
            { "ReLU", new ReLU<int>() },
            { "SoftMax", new SoftMax<int>() },
        };
    }

private:
    static std::unordered_map<std::string, ActivationFunc<float>*>
    m_floatActivationMap;
    static std::unordered_map<std::string, ActivationFunc<int>*>
    m_integerActivationMap;
};
}
#endif

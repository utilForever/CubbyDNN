// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_LOSSFUNCTIONWRAPPER_HPP
#define TAKION_COMPUTE_LOSSFUNCTIONWRAPPER_HPP

#include <Takion/Computations/LossFunctions/LossFunctions.hpp>
#include <memory>
#include <unordered_map>

namespace Takion::Compute
{
template <typename T>
class LossFunctionWrapper
{
public:
    const static std::unique_ptr<BaseLoss<float>>& GetFloatLoss(
        const std::string& name)
    {
        return m_floatLossMap.at(name);
    }

    static void Initialize()
    {
        m_floatLossMap["MSE"] = std::make_unique<MSE<T>>();
    }

private:
    static std::unordered_map<std::string,
                              std::unique_ptr<BaseLoss<T>>>
    m_floatLossMap;
};

template <typename T>
std::unordered_map<std::string, std::unique_ptr<BaseLoss<T>>>
LossFunctionWrapper<T>::m_floatLossMap;
}
#endif

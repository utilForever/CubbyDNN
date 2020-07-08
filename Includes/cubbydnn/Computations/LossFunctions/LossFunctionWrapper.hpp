// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTE_LOSSFUNCTIONWRAPPER_HPP
#define CUBBYDNN_COMPUTE_LOSSFUNCTIONWRAPPER_HPP
#include <cubbydnn/Computations/LossFunctions/LossFunctions.hpp>
#include <memory>
#include <unordered_map>

namespace CubbyDNN::Compute
{
class LossFunctionWrapper
{
 public:
    const static std::unique_ptr<Loss<float>>& GetFloatLoss(
        const std::string& name)
    {
        return m_floatLossMap.at(name);
    }
 
    const static std::unique_ptr<Loss<int>>& GetIntegerLoss(
        const std::string& name)
    {
        return m_integerLossMap.at(name);
    }
 
    static void Initialize()
    {
        m_floatLossMap = { { "MSE", std::make_unique<MSE<float>>() } };
        m_integerLossMap = { { "MSE", std::make_unique<MSE<int>>() } };
    }
 
 private:
    static std::unordered_map<std::string,
                              std::unique_ptr<Loss<float>>>
        m_floatLossMap;
 
    static std::unordered_map<std::string, std::unique_ptr<Loss<int>>>
        m_integerLossMap;

};
}
#endif
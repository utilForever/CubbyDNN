// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_ACTIVATION_HPP
#define CUBBYDNN_ACTIVATION_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Computations/Activations/ActivationFunc.hpp>
#include <cubbydnn/Units/UnitMetadata.hpp>


namespace CubbyDNN::Graph
{
class ActivationUnit : public ComputableUnit
{
public:
    ActivationUnit(UnitId unitId, NumberSystem numberSystem,
                   Tensor forwardInput, std::vector<Tensor> backwardInputVector,
                   Tensor forwardOutput, Tensor backwardOutput,
                   Tensor backwardTemp,
                   std::unique_ptr<Compute::ActivationFunc> activationFunc);


    ActivationUnit(const ActivationUnit& activationUnit) = delete;
    ActivationUnit(ActivationUnit&& activationUnit) noexcept;
    ActivationUnit& operator=(const ActivationUnit& activationUnit) = delete;
    ActivationUnit& operator=(ActivationUnit&& activationUnit) noexcept;
    ~ActivationUnit() = default;

    static ActivationUnit CreateUnit(const UnitMetaData& unitMetaData,
                                     std::unique_ptr<Compute::ActivationFunc>
                                     activationFunc);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

private:
    std::unique_ptr<Compute::ActivationFunc> m_activationFunc;
    Tensor m_backwardTemp;
};
} // namespace CubbyDNN::Graph

#endif

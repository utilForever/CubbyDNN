// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_DENSE_HPP
#define CUBBYDNN_DENSE_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Units/UnitMetadata.hpp>
#include <cubbydnn/Computations/Initializers/InitializerType.hpp>


namespace CubbyDNN::Graph
{
class DenseUnit : public ComputableUnit
{
public:
    DenseUnit(UnitId unitId, NumberSystem numberSystem, Tensor forwardInput,
              std::vector<Tensor> backwardInputVector, Tensor forwardOutput,
              Tensor backwardOutput,
              Tensor weight, Tensor bias,
              Tensor weightTranspose);
    ~DenseUnit() = default;

    DenseUnit(const DenseUnit& dense) = delete;
    DenseUnit(DenseUnit&& dense) noexcept;
    DenseUnit& operator=(const DenseUnit& dens) = delete;
    DenseUnit& operator=(DenseUnit&& dense) noexcept;

    static DenseUnit CreateUnit(const UnitMetaData& unitMetaData,
                                float dropout);

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;

    void Initialize(const std::vector<std::unique_ptr<Initializer>>&
        initializerVector) override;

private:
    Tensor m_kernel;
    Tensor m_bias;
    Tensor m_transposedKernel;
};
} // namespace CubbyDNN

#endif

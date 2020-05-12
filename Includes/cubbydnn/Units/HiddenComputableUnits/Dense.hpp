// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_DENSE_HPP
#define CUBBYDNN_DENSE_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Computations/Initializers/InitializerType.hpp>


namespace CubbyDNN::Graph
{
class DenseUnit : public ComputableUnit
{
public:
    DenseUnit(UnitId unitId, NumberSystem numberSystem, Tensor forwardInput,
              std::vector<Tensor> backwardInputVector, Tensor forwardOutput,
              Tensor backwardOutput,
              Shape weightShape, Shape biasShape,
              float dropoutRate, std::size_t padSize);
    ~DenseUnit() = default;

    DenseUnit(const DenseUnit& dense) = delete;
    DenseUnit(DenseUnit&& dense) noexcept;
    DenseUnit& operator=(const DenseUnit& dens) = delete;
    DenseUnit& operator=(DenseUnit&& dense) noexcept;

    void Forward() override;

    void Backward() override;

private:
    Tensor m_kernel;
    Tensor m_bias;
    Tensor m_temp;
    Tensor m_transposedKernel;
    float m_dropoutRate;
};
} // namespace CubbyDNN

#endif

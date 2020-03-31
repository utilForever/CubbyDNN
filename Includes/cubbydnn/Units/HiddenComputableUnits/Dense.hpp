// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_DENSE_HPP
#define CUBBYDNN_DENSE_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
class DenseUnit : public ComputableUnit
{
public:
    DenseUnit(UnitId unitId, Shape input, Shape weightShape, Shape biasShape,
              Shape output, NumberSystem numberSystem,
              InitializerType kernelInitializer,
              InitializerType biasInitializer, Activation activation,
              float dropoutRate);
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
    InitializerType m_kernelInitializer;
    InitializerType m_biasInitializer;
    Activation m_activation;
    float m_dropoutRate;
};
} // namespace CubbyDNN

#endif

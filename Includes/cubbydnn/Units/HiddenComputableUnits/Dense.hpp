// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_DENSE_HPP
#define CUBBYDNN_DENSE_HPP

#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>

namespace CubbyDNN
{
class DenseUnit : HiddenUnit
{
 public:
    DenseUnit(TensorInfo input, TensorInfo weight, TensorInfo bias,
                 TensorInfo output,  std::size_t numUnits, std::size_t numberOfOutputs);
    ~DenseUnit() = default;

    DenseUnit(const DenseUnit& dense) = delete;
    DenseUnit(DenseUnit&& dense) noexcept;
    DenseUnit& operator=(const DenseUnit& dens) = delete;
    DenseUnit& operator=(DenseUnit&& dense) noexcept;

    void Forward();

    void Backward();

 private:
    std::size_t m_numUnits;
    Tensor m_temp = Tensor(nullptr, TensorInfo());
};
}  // namespace CubbyDNN

#endif
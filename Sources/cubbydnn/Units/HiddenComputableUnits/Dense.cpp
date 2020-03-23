// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbyDnn/Units/HiddenComputableUnits/Dense.hpp>

namespace CubbyDNN
{
DenseUnit::DenseUnit(TensorInfo input, TensorInfo weight, TensorInfo bias,
                     TensorInfo output)
    : HiddenUnit(
          { input, weight, bias }, std::move(output))

{
    Shape tempShape(weight.GetShape());
    tempShape.SetCol(1);
    m_temp = AllocateTensor(TensorInfo(tempShape, weight.GetNumberSystem()));
}

DenseUnit::DenseUnit(DenseUnit&& dense) noexcept
    : HiddenUnit(std::move(dense))
{
}

DenseUnit& DenseUnit::operator=(DenseUnit&& dense) noexcept
{
    m_temp = std::move(dense.m_temp);
    HiddenUnit::operator=(std::move(dense));
    return *this;
}

void DenseUnit::Forward()
{
    Tensor& input = m_inputForwardTensorVector.at(0);
    Tensor& weight = m_inputForwardTensorVector.at(1);
    Tensor& bias = m_inputForwardTensorVector.at(2);
    Tensor& output = m_outputForwardTensor;

    m_tensorOperation->Multiply(weight, input, m_temp);
    m_tensorOperation->Add(m_temp, bias, output);
}

void DenseUnit::Backward()
{
    // TODO : Create separate inputs and outputs for back propagation
    // Tensor& delta = m_inputTensorVector.at(0);
    // Tensor& weight = m_inputTensorVector.at(1);
    // Tensor& input = m_inputTensorVector.at(0); // (W(l+1) & delta(l + 1)
    // m_tensorOperation->Multiply(weight, delta, m_temp);
}
} // namespace CubbyDNN

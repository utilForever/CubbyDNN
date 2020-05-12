// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Computations/Initializers/InitializerType.hpp>
#include <cubbydnn/Computations/TensorOperations/NaiveOperations.hpp>

namespace CubbyDNN::Graph
{
DenseUnit::DenseUnit(UnitId unitId, Shape input, Shape weightShape,
                     Shape biasShape,
                     Shape output,
                     NumberSystem numberSystem,
                     std::unique_ptr<Initializer> kernelInitializer,
                     std::unique_ptr<Initializer> biasInitializer,
                     Activation activation,
                     float dropoutRate, std::size_t padSize)
    : ComputableUnit(unitId,
                     { std::move(input) }, std::move(output), numberSystem),
      m_kernel(CreateTensor(weightShape, numberSystem, padSize)),
      m_bias(CreateTensor(biasShape, numberSystem, padSize)),
      m_kernelInitializer(std::move(kernelInitializer)),
      m_biasInitializer(std::move(biasInitializer)),
      m_activation(activation),
      m_dropoutRate(dropoutRate)

{
    //TODO : make this selectable
    m_kernelInitializer->Initialize(m_kernel, m_numberSystem);
    m_biasInitializer->Initialize(m_bias, m_numberSystem);
}

DenseUnit::DenseUnit(DenseUnit&& dense) noexcept
    : ComputableUnit(std::move(dense)),
      m_kernel(std::move(dense.m_kernel)),
      m_bias(std::move(dense.m_bias)),
      m_kernelInitializer(std::move(dense.m_kernelInitializer)),
      m_biasInitializer(std::move(dense.m_biasInitializer)),
      m_activation(dense.m_activation),
      m_dropoutRate(dense.m_dropoutRate)
{
}

DenseUnit& DenseUnit::operator=(DenseUnit&& dense) noexcept
{
    m_kernel = std::move(dense.m_kernel);
    m_bias = std::move(dense.m_bias);
    m_kernelInitializer = std::move(dense.m_kernelInitializer);
    m_biasInitializer = std::move(dense.m_biasInitializer);
    m_activation = dense.m_activation;
    m_dropoutRate = dense.m_dropoutRate;
    ComputableUnit::operator=(std::move(dense));
    return *this;
}

void DenseUnit::Forward()
{
    Tensor& input = ForwardInputVector.at(0);
    Tensor& weight = ForwardInputVector.at(1);
    Tensor& bias = ForwardInputVector.at(2);
    Tensor& output = m_fowrardOutput;

    Native::Multiply(weight, input, m_kernel);
    Native::Add(m_kernel, bias, output);
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

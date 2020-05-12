// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Computations/Initializers/InitializerType.hpp>
#include <cubbydnn/Computations/TensorOperations/NaiveOperations.hpp>

namespace CubbyDNN::Graph
{
DenseUnit::DenseUnit(UnitId unitId, NumberSystem numberSystem,
                     Tensor forwardInput,
                     std::vector<Tensor> backwardInputVector,
                     Tensor forwardOutput, Tensor backwardOutput,
                     Shape weightShape, Shape biasShape, float dropoutRate,
                     std::size_t padSize)
    : ComputableUnit(std::move(unitId), numberSystem,
                     { std::move(forwardInput) },
                     std::move(backwardInputVector),
                     std::move(forwardOutput), std::move(backwardOutput)),
      m_kernel(CreateTensor(weightShape, numberSystem, padSize)),
      m_bias(CreateTensor(biasShape, numberSystem, padSize)),
      m_temp(CreateTensor(forwardOutput.TensorShape, numberSystem, padSize)),
      m_transposedKernel(CreateTensor(weightShape, numberSystem, padSize)),
      m_dropoutRate(dropoutRate)
{
}

DenseUnit::DenseUnit(DenseUnit&& dense) noexcept
    : ComputableUnit(std::move(dense)),
      m_kernel(std::move(dense.m_kernel)),
      m_bias(std::move(dense.m_bias)),
      m_temp(std::move(dense.m_temp)),
      m_transposedKernel(std::move(dense.m_transposedKernel)),
      m_dropoutRate(dense.m_dropoutRate)
{
}

DenseUnit& DenseUnit::operator=(DenseUnit&& dense) noexcept
{
    m_kernel = std::move(dense.m_kernel);
    m_bias = std::move(dense.m_bias);
    m_temp = std::move(dense.m_temp);
    m_transposedKernel = std::move(dense.m_transposedKernel);
    m_dropoutRate = dense.m_dropoutRate;
    ComputableUnit::operator=(std::move(dense));
    return *this;
}

void DenseUnit::Forward()
{
    Tensor& input = ForwardInputVector.at(0);

    Native::Multiply(m_kernel, input, m_temp);
    Native::Add(m_temp, m_bias, ForwardOutput);
}

void DenseUnit::Backward()
{
    Tensor& delta = BackwardInputVector.at(0);

    Native::Transpose(m_kernel, m_transposedKernel);
    Native::Multiply(m_transposedKernel, delta, BackwardOutput);

    // TODO : Update kernel using gradient updater
}
} // namespace CubbyDNN

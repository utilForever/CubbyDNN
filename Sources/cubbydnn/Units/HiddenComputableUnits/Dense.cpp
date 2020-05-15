// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Computations/TensorOperations/NaiveOperations.hpp>

namespace CubbyDNN::Graph
{
DenseUnit::DenseUnit(UnitId unitId, NumberSystem numberSystem,
                     Tensor forwardInput,
                     std::vector<Tensor> backwardInputVector,
                     Tensor forwardOutput, Tensor backwardOutput,
                     Tensor weight, Tensor bias,
                     Tensor weightTranspose)
    : ComputableUnit(std::move(unitId), numberSystem,
                     { std::move(forwardInput) },
                     std::move(backwardInputVector), std::move(forwardOutput),
                     { std::move(backwardOutput) }),
      m_kernel(std::move(weight)),
      m_bias(std::move(bias)),
      m_transposedKernel(std::move(weightTranspose))
{
}

DenseUnit::DenseUnit(DenseUnit&& dense) noexcept
    : ComputableUnit(std::move(dense)),
      m_kernel(std::move(dense.m_kernel)),
      m_bias(std::move(dense.m_bias)),
      m_transposedKernel(std::move(dense.m_transposedKernel))
{
}

DenseUnit& DenseUnit::operator=(DenseUnit&& dense) noexcept
{
    m_kernel = std::move(dense.m_kernel);
    m_bias = std::move(dense.m_bias);
    m_transposedKernel = std::move(dense.m_transposedKernel);
    ComputableUnit::operator=(std::move(dense));
    return *this;
}

DenseUnit DenseUnit::CreateUnit(const UnitMetaData& unitMetaData, float dropout)
{
    const auto unitId = unitMetaData.Id();

    auto forwardInputTensor =
        Tensor::CreateTensor(unitMetaData.InputShapeVector().at(0),
                             unitMetaData.NumericType, unitMetaData.Device);
    std::vector<Tensor> backwardInputVector;

    backwardInputVector.reserve(unitMetaData.OutputUnitVector().size());
    for (std::size_t i = 0; i < unitMetaData.OutputUnitVector().size(); ++i)
    {
        auto tensor = Tensor::CreateTensor(unitMetaData.OutputShape(),
                                           unitMetaData.NumericType,
                                           unitMetaData.Device);
        backwardInputVector.emplace_back(std::move(tensor));
    }

    auto forwardOutputTensor =
        Tensor::CreateTensor(unitMetaData.OutputShape(),
                             unitMetaData.NumericType, unitMetaData.Device);

    auto backwardOutputTensor =
        Tensor::CreateTensor(unitMetaData.InputShapeVector().at(0),
                             unitMetaData.NumericType, unitMetaData.Device);

    auto weightShape = unitMetaData.InternalVariableShapeVector().at(0);
    auto biasShape = unitMetaData.InternalVariableShapeVector().at(1);

    auto weightTensor =
        Tensor::CreateTensor(weightShape, unitMetaData.NumericType,
                             unitMetaData.Device, unitMetaData.PadSize);

    auto biasTensor =
        Tensor::CreateTensor(biasShape, unitMetaData.NumericType,
                             unitMetaData.Device, unitMetaData.PadSize);

    auto weightTransposeTensor = Tensor::CreateTensor(
        weightShape.Transpose(), unitMetaData.NumericType,
        unitMetaData.Device, unitMetaData.PadSize);

    auto denseUnit = DenseUnit(
        unitId, unitMetaData.NumericType, std::move(forwardInputTensor),
        std::move(backwardInputVector), std::move(forwardOutputTensor),
        std::move(backwardOutputTensor), std::move(weightTensor),
        std::move(biasTensor),
        std::move(weightTransposeTensor));

    return std::move(denseUnit);
}


void DenseUnit::Forward()
{
    Tensor& input = ForwardInputVector.at(0);

    Native::Multiply(m_kernel, input, ForwardOutput);
    Native::Add(ForwardOutput, m_bias, ForwardOutput);
}

void DenseUnit::AsyncForward(std::promise<bool> promise)
{
    Tensor& input = ForwardInputVector.at(0);

    Native::Multiply(m_kernel, input, ForwardOutput);
    Native::Add(ForwardOutput, m_bias, ForwardOutput);
    promise.set_value(true);
}


void DenseUnit::Backward()
{
    Tensor& delta = BackwardInputVector.at(0);

    Native::Transpose(m_kernel, m_transposedKernel);
    Native::Multiply(m_transposedKernel, delta, BackwardOutputVector.at(0));

    // TODO : Update kernel using gradient optimizer
}

void DenseUnit::AsyncBackward(std::promise<bool> promise)
{
    Tensor& delta = BackwardInputVector.at(0);

    Native::Transpose(m_kernel, m_transposedKernel);
    Native::Multiply(m_transposedKernel, delta, BackwardOutputVector.at(0));

    // TODO : Update kernel using gradient optimizer
    promise.set_value(true);
}

void DenseUnit::Initialize(
    const std::vector<std::unique_ptr<Initializer>>& initializerVector)
{
    initializerVector.at(0)->Initialize(m_kernel, m_numericType);
    initializerVector.at(1)->Initialize(m_bias, m_numericType);

    const Zeros zeroInitializer;
    zeroInitializer.Initialize(m_transposedKernel, m_numericType);
}
} // namespace CubbyDNN

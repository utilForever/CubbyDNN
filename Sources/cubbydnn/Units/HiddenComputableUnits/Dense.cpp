// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Computations/TensorOperations/Computations.hpp>

namespace CubbyDNN::Graph
{
static const int weightIdx = 0;
static const int biasIdx = 1;

DenseUnit::DenseUnit(UnitId unitId, NumberSystem numberSystem,
                     Tensor forwardInput,
                     std::vector<Tensor> backwardInputVector,
                     Tensor forwardOutput, Tensor backwardOutput,
                     std::vector<Tensor> trainableUnit,
                     std::unique_ptr<Computation::Optimizer> optimizer,
                     Tensor weightTranspose)
    : ComputableUnit(std::move(unitId), numberSystem,
                     { std::move(forwardInput) },
                     std::move(backwardInputVector), std::move(forwardOutput),
                     { std::move(backwardOutput) }),
      TrainableUnit(std::move(trainableUnit), std::move(optimizer)),
      m_transposedWeight(std::move(weightTranspose))
{
}

DenseUnit::DenseUnit(DenseUnit&& denseUnit) noexcept
    : ComputableUnit(std::move(denseUnit)),
      TrainableUnit(std::move(denseUnit)),
      m_transposedWeight(std::move(denseUnit.m_transposedWeight))
{
}

DenseUnit& DenseUnit::operator=(DenseUnit&& denseUnit) noexcept
{
    ComputableUnit::operator=(std::move(denseUnit));
    TrainableUnit::operator=(std::move(denseUnit));
    m_transposedWeight = std::move(denseUnit.m_transposedWeight);

    return *this;
}

DenseUnit DenseUnit::CreateUnit(const UnitMetaData& unitMetaData,
                                std::unique_ptr<Computation::Optimizer>
                                optimizer)
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

    auto weightShape = unitMetaData.InternalVariableShapeVector().at(weightIdx);
    auto biasShape = unitMetaData.InternalVariableShapeVector().at(biasIdx);

    auto weightTensor =
        Tensor::CreateTensor(weightShape, unitMetaData.NumericType,
                             unitMetaData.Device);
    const auto& weightInitializer = unitMetaData
                                    .InitializerVector().at(weightIdx);
    weightInitializer->Initialize(weightTensor);

    auto biasTensor =
        Tensor::CreateTensor(biasShape, unitMetaData.NumericType,
                             unitMetaData.Device);
    const auto& biasInitializer = unitMetaData.InitializerVector().at(biasIdx);
    biasInitializer->Initialize(biasTensor);

    auto weightTransposeTensor = Tensor::CreateTensor(
        weightShape.Transpose(), unitMetaData.NumericType,
        unitMetaData.Device);

    auto denseUnit = DenseUnit(
        unitId, unitMetaData.NumericType, std::move(forwardInputTensor),
        std::move(backwardInputVector), std::move(forwardOutputTensor),
        std::move(backwardOutputTensor),
        { std::move(weightTensor), std::move(biasTensor) },
        std::move(optimizer),
        std::move(weightTransposeTensor));

    return denseUnit;
}


void DenseUnit::Forward()
{
    Tensor& input = ForwardInputVector.at(0);

    Compute::Multiply(m_trainableTensorMap.at(weightIdx), input,
                              ForwardOutput);
    Compute::Add(ForwardOutput, m_trainableTensorMap.at(biasIdx),
                         ForwardOutput);
}

void DenseUnit::AsyncForward(std::promise<bool> promise)
{
    Tensor& input = ForwardInputVector.at(0);

    Compute::Multiply(m_trainableTensorMap.at(weightIdx), input,
                              ForwardOutput);
    Compute::Add(ForwardOutput, m_trainableTensorMap.at(biasIdx),
                         ForwardOutput);
    promise.set_value(true);
}

void DenseUnit::Backward()
{
    auto& weight = m_trainableTensorMap.at(weightIdx);
    auto& bias = m_trainableTensorMap.at(biasIdx);

    Tensor& delta = BackwardInputVector.at(0);

    Compute::Transpose(weight, m_transposedWeight);
    Compute::Multiply(m_transposedWeight, delta,
                              BackwardOutputVector.at(0));

    m_optimizer->Optimize(weight);
    m_optimizer->Optimize(bias);
}

void DenseUnit::AsyncBackward(std::promise<bool> promise)
{
    auto& weight = m_trainableTensorMap.at(weightIdx);
    auto& bias = m_trainableTensorMap.at(biasIdx);
    Tensor& delta = BackwardInputVector.at(0);

    Compute::Transpose(weight, bias);
    Compute::Multiply(m_transposedWeight, delta,
                              BackwardOutputVector.at(0));

    m_optimizer->Optimize(weight);
    m_optimizer->Optimize(bias);

    promise.set_value(true);
}
} // namespace CubbyDNN

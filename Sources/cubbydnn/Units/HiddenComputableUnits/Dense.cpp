// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Computations/TensorOperations/Computations.hpp>

namespace CubbyDNN::Graph
{
DenseUnit::DenseUnit(UnitId unitId, NumberSystem numberSystem,
                     Tensor forwardInput,
                     std::vector<Tensor> backwardInputVector,
                     Tensor forwardOutput, Tensor backwardOutput,
                     std::unordered_map<std::string, Tensor> trainableUnit,
                     std::unique_ptr<Compute::Optimizer> optimizer)
    : ComputableUnit(std::move(unitId), numberSystem,
                     { std::move(forwardInput) },
                     std::move(backwardInputVector), std::move(forwardOutput),
                     { std::move(backwardOutput) }),
      TrainableUnit(std::move(trainableUnit), std::move(optimizer))
{
}

DenseUnit::DenseUnit(DenseUnit&& denseUnit) noexcept
    : ComputableUnit(std::move(denseUnit)),
      TrainableUnit(std::move(denseUnit))
{
}

DenseUnit& DenseUnit::operator=(DenseUnit&& denseUnit) noexcept
{
    ComputableUnit::operator=(std::move(denseUnit));
    TrainableUnit::operator=(std::move(denseUnit));

    return *this;
}

DenseUnit DenseUnit::CreateUnit(const UnitMetaData& unitMetaData,
                                std::unique_ptr<Compute::Optimizer>
                                optimizer)
{
    const auto unitId = unitMetaData.Id();

    Tensor forwardInputTensor(unitMetaData.InputShapeVector().at(0),
                              unitMetaData.Device,
                              unitMetaData.NumericType);

    std::vector<Tensor> backwardInputVector;
    backwardInputVector.reserve(unitMetaData.OutputUnitVector().size());
    for (std::size_t i = 0; i < unitMetaData.OutputUnitVector().size(); ++i)
    {
        Tensor tensor(unitMetaData.OutputShape(),
                      unitMetaData.Device, unitMetaData.NumericType);
        backwardInputVector.emplace_back(std::move(tensor));
    }

    Tensor forwardOutputTensor(unitMetaData.OutputShape(),
                               unitMetaData.Device, unitMetaData.NumericType);

    Tensor backwardOutputTensor(unitMetaData.InputShapeVector().at(0),
                                unitMetaData.Device, unitMetaData.NumericType);

    auto weightShape = unitMetaData.GetShape("weight");
    auto biasShape = unitMetaData.GetShape("bias");

    Tensor weightTensor(weightShape,
                        unitMetaData.Device, unitMetaData.NumericType);
    const auto& weightInitializer = unitMetaData.GetInitializer("weight");
    weightInitializer->Initialize(weightTensor);

    Tensor weightDelta(weightShape, unitMetaData.Device,
                       unitMetaData.NumericType);

    Tensor biasTensor(biasShape, unitMetaData.Device,
                      unitMetaData.NumericType);
    const auto& biasInitializer = unitMetaData.GetInitializer("bias");
    biasInitializer->Initialize(biasTensor);

    Tensor transposedWeight(weightShape.Transpose(),
                            unitMetaData.Device, unitMetaData.NumericType);

    Shape batchMeanDeltaShape{ backwardInputVector.at(0).TensorShape[0],
                               backwardInputVector.at(0).TensorShape[1] };

    Tensor batchMeanDelta(batchMeanDeltaShape, unitMetaData.Device,
                          unitMetaData.NumericType);

    auto denseUnit = DenseUnit(
        unitId, unitMetaData.NumericType, std::move(forwardInputTensor),
        { std::move(backwardInputVector) }, std::move(forwardOutputTensor),
        std::move(backwardOutputTensor),
        { { "weight", std::move(weightTensor) },
          { "bias", std::move(biasTensor) },
          { "weightTranspose", std::move(transposedWeight) },
          { "batchMeanDelta", std::move(batchMeanDelta) } },
        std::move(optimizer));

    return denseUnit;
}


void DenseUnit::Forward()
{
    Tensor& input = ForwardInputVector.at(0);

    Compute::Multiply(m_trainableTensorMap["weight"], input,
                      ForwardOutput);
    Compute::Add(ForwardOutput, m_trainableTensorMap["bias"]);
}

void DenseUnit::AsyncForward(std::promise<bool> promise)
{
    Tensor& input = ForwardInputVector.at(0);

    Compute::Multiply(m_trainableTensorMap["weight"], input,
                      ForwardOutput);
    Compute::Add(ForwardOutput, m_trainableTensorMap["bias"]);
    promise.set_value(true);
}

//TODO : Make it receive multiple backward inputs
void DenseUnit::Backward()
{
    auto& weight = m_trainableTensorMap["weight"];
    auto& bias = m_trainableTensorMap["bias"];
    auto& weightUpdate = m_trainableTensorMap["weightUpdate"];
    auto& biasUpdate = m_trainableTensorMap["biasUpdate"];
    const auto batchSize = weight.TensorShape.NumCols();

    auto& previousForwardInput = ForwardInputVector.at(0);
    auto& backwardOutput = BackwardOutputVector.at(0);

    Tensor& delta = m_trainableTensorMap["delta"];
    for (Tensor& gradient : BackwardInputVector)
    {
        Compute::Add(delta, gradient);
    }

    Compute::ScalarMul(delta, 1.0f / BackwardInputVector.size());
    Compute::Multiply(weight, delta,
                      backwardOutput, true, false);

    Compute::Multiply(delta, previousForwardInput, weightUpdate, false, true);
    Compute::ScalarMul(weightUpdate, 1.0f / batchSize);

    Compute::BatchMean(delta, biasUpdate, delta.TensorShape.Dim() - 2);

    m_optimizer->Optimize(weight, weightUpdate);
    m_optimizer->Optimize(bias, biasUpdate);
}

void DenseUnit::AsyncBackward(std::promise<bool> promise)
{
    auto& weight = m_trainableTensorMap["weight"];
    auto& bias = m_trainableTensorMap["bias"];
    auto& weightUpdate = m_trainableTensorMap["weightUpdate"];
    auto& biasUpdate = m_trainableTensorMap["biasUpdate"];
    const auto batchSize = weight.TensorShape.NumCols();

    auto& previousForwardInput = ForwardInputVector.at(0);
    auto& backwardOutput = BackwardOutputVector.at(0);

    Tensor& delta = m_trainableTensorMap["delta"];
    for (Tensor& gradient : BackwardInputVector)
    {
        Compute::Add(delta, gradient);
    }

    Compute::ScalarMul(delta, 1.0f / BackwardInputVector.size());
    Compute::Multiply(weight, delta, backwardOutput, true, false);

    Compute::Multiply(delta, previousForwardInput, weightUpdate, false, true);
    Compute::ScalarMul(weightUpdate, 1.0f / batchSize);

    Compute::BatchMean(delta, biasUpdate, delta.TensorShape.Dim() - 2);

    m_optimizer->Optimize(weight, weightUpdate);
    m_optimizer->Optimize(bias, biasUpdate);

    promise.set_value(true);
}
} // namespace CubbyDNN

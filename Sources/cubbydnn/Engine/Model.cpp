// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Engine/Model.hpp>

namespace CubbyDNN::Graph
{
Model::Model(NumberSystem numericType)
    : m_numericType(numericType)
{
}

UnitId Model::DataLoader(const Shape& shape, const std::string& name,
                         Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Source, "DataLoader"), m_id++,
                          name };
    UnitMetaData unitMetaData(subjectUnitId, {}, {},
                              {}, shape, {},
                              m_numericType,
                              std::move(device));
    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

UnitId Model::Dense(const UnitId& input, std::size_t units,
                    std::unique_ptr<Initializer> weightInitializer,
                    std::unique_ptr<Initializer> biasInitializer,
                    const std::string& name,
                    Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "Dense"), m_id++,
                          name };
    const auto previousOutputShape = m_unitManager.GetUnitOutputShape(input);
    auto weightShape = previousOutputShape;
    weightShape.SetNumRows(units);
    auto biasShape = weightShape;
    biasShape.SetNumCols(1);
    biasShape.SetNumRows(units);

    auto outputShape = previousOutputShape;
    outputShape.SetNumRows(units);

    std::unordered_map<std::string, std::unique_ptr<Initializer>>
        initializerMap;

    initializerMap["weight"] = std::move(weightInitializer);
    initializerMap["bias"] = std::move(biasInitializer);

    UnitMetaData unitMetaData(subjectUnitId,
                              { { "weight", weightShape },
                                { "bias", biasShape } },
                              std::move(initializerMap),
                              { { "input", previousOutputShape } }, outputShape,
                              { { "input", input } }, m_numericType,
                              std::move(device));

    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

UnitId Model::Activation(const UnitId& input, const std::string& activationName,
                         const std::string& name, Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "Activation"), m_id++,
                          name };
    const auto previousOutputShape = m_unitManager.GetUnitOutputShape(input);

    UnitMetaData unitMetaData(subjectUnitId, {}, {},
                              { { "input", previousOutputShape } },
                              { previousOutputShape }, { { "input", input } },
                              m_numericType,
                              std::move(device),
                              Parameter(
                                  { { "activationName", activationName } }));

    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

UnitId Model::Constant(Tensor tensor, const std::string& name)
{
    if (tensor.NumericType != m_numericType)
        throw std::invalid_argument(
            "Numeric type of given tensor and graph must be identical");
    UnitId subjectUnitId{ UnitType(UnitBaseType::Source, "Constant"), m_id++,
                          name };
    UnitMetaData unitMetaData(subjectUnitId, {}, {},
                              {}, tensor.TensorShape, {},
                              tensor.NumericType, tensor.Device);
    unitMetaData.AddInternalTensor("constant", std::move(tensor));
    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

void Model::Compile(const std::string& optimizer,
                    Parameter optimizerParams) noexcept
{
    m_unitManager.Compile(optimizer, optimizerParams);
}

UnitId Model::Loss(const UnitId& prediction, const UnitId& label,
                   std::string lossType,
                   const std::string& name, Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Sink, "Loss"), m_id++, name };
    const auto predictionShape = m_unitManager.GetUnitOutputShape(prediction);
    const auto labelShape = m_unitManager.GetUnitOutputShape(label);

    UnitMetaData unitMetaData(
        subjectUnitId, {}, {},
        { { "prediction", predictionShape }, { "label", labelShape } }, Shape(),
        { { "prediction", prediction }, { "label", label } }, m_numericType,
        std::move(device), Parameter({ { "lossType", lossType } }));
    m_unitManager.AppendUnit(std::move(unitMetaData));

    return subjectUnitId;
}

// UnitId Model::Loss(const UnitId& prediction, const UnitId& label,
//                    const std::string& name, Compute::Device device,
//                    std::unique_ptr<Compute::BaseLoss<float>> customLoss)
// {
//     UnitId subjectUnitId{ UnitType(UnitBaseType::Sink, "Loss"), m_id++, name };
//     const auto predictionShape = m_unitManager.GetUnitOutputShape(prediction);
//     const auto labelShape = m_unitManager.GetUnitOutputShape(label);
//
//     UnitMetaData unitMetaData(
//         subjectUnitId, {}, {},
//         { { "prediction", predictionShape }, { "label", labelShape } }, Shape(),
//         { { "prediction", prediction }, { "label", label } }, m_numericType,
//         std::move(device));
//
//     return subjectUnitId;
// }


void Model::Train(std::size_t epochs, bool async)
{
    if (async)
    {
        for (std::size_t i = 0; i < epochs; ++i)
        {
            m_unitManager.AsyncForward(i);
            m_unitManager.AsyncBackward(i);
        }
        return;
    }
    for (std::size_t i = 0; i < epochs; ++i)
    {
        m_unitManager.Forward(i);
        m_unitManager.Backward(i);
    }
}
} // namespace CubbyDNN

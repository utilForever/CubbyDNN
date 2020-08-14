// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_FRONTEND_MODEL_HPP
#define TAKION_FRONTEND_MODEL_HPP

#include <Takion/FrontEnd/ModelDecl.hpp>

namespace Takion::FrontEnd
{
template <typename T>
Model<T>::Model(Compute::Device device, std::size_t batchSize)
    : m_device(device),
      m_unitManager(Graph::UnitManager<T>(batchSize)),
      m_batchSize(batchSize)
{
}

template <typename T>
void Model<T>::SetDevice(Compute::Device device)
{
    m_device = device;
}

template <typename T>
AbsTensor<T> Model<T>::Constant(const Shape& shape, std::vector<T> data,
                                std::string name)
{
    if (data.size() != shape.Size())
    {
        const std::string errorMessage =
            std::string("Constant - ") + name +
            " Given data and shape's size should be identical" +
            " shape : " + shape.ToString() +
            " dataSize : " + std::to_string(data.Size());
        throw std::invalid_argument(errorMessage);
    }

    const UnitId subjectUnitId{ UnitType(UnitBaseType::Source, "Constant"),
                                m_id++, std::move(name) };

    std::unordered_map<std::string, std::unique_ptr<Compute::Initializer<T>>>
        initializerMap;
    initializerMap["vectorInitializer"] = Compute::VectorInitializer<T>(data);

    UnitMetaData<T> unitMetaData(subjectUnitId, m_batchSize, {}, initializerMap,
                                 {}, shape,
                                 {}, m_device);
    m_unitManager.AppendUnit(unitMetaData);

    return subjectUnitId;
}


template <typename T>
AbsTensor<T> Model<T>::Dense(AbsTensor<T> source, unsigned numUnits,
                             std::unique_ptr<Compute::Initializer<T>>
                             weightInitializer,
                             std::unique_ptr<Compute::Initializer<T>>
                             biasInitializer, std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "Dense"), m_id++,
                                std::move(name) };

    const auto prevUnitId = source.GetPrevOutput();
    const auto prevOutputShape =
        m_unitManager.GetUnitOutputShape(prevUnitId);
    const auto inputShape = source.GetShape();

    const Shape weightShape({ prevOutputShape.NumCol(), numUnits });
    const Shape biasShape({ 1, numUnits });
    const Shape outputShape({ 1, numUnits });

    std::unordered_map<std::string, std::unique_ptr<Compute::Initializer<T>>>
        initializerMap;

    initializerMap["weight"] = std::move(weightInitializer);
    initializerMap["bias"] = std::move(biasInitializer);

    UnitMetaData<T> unitMetaData(
        subjectUnitId, m_batchSize,
        { { "weight", weightShape }, { "bias", biasShape } },
        std::move(initializerMap), { { "input", prevOutputShape } },
        outputShape,
        { { "input", prevUnitId } }, m_device);

    m_unitManager.AppendUnit(std::move(unitMetaData));

    return AbsTensor<T>(outputShape, subjectUnitId);
}

template <typename T>
AbsTensor<T> Model<T>::ReLU(AbsTensor<T> source, std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "Dense"), m_id++,
                                std::move(name) };

    const auto prevUnitId = source.GetPrevOutput();
    const auto shape = source.GetShape();

    std::unordered_map<std::string, std::unique_ptr<Compute::Initializer<T>>>
        initializerMap;

    UnitMetaData<T> unitMetaData(
        subjectUnitId, m_batchSize, {}, {}, { { "input", shape } },
        shape, { { "input", prevUnitId } }, m_device);

    m_unitManager.AppendUnit(std::move(unitMetaData));

    return AbsTensor<T>(shape, subjectUnitId);
}

template <typename T>
AbsTensor<T> Model<T>::SoftMax(AbsTensor<T> source, std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "Dense"), m_id++,
                                std::move(name) };

    const auto prevUnitId = source.GetPrevOutput();
    const auto shape = source.GetShape();

    std::unordered_map<std::string, std::unique_ptr<Compute::Initializer<T>>>
        initializerMap;

    UnitMetaData<T> unitMetaData(subjectUnitId, m_batchSize, {}, {},
                                 { { "input", shape } }, shape,
                                 { { "input", prevUnitId } }, m_device);

    m_unitManager.AppendUnit(std::move(unitMetaData));

    return AbsTensor<T>(shape, subjectUnitId);
}

template <typename T>
void Model<T>::MSE(AbsTensor<T> prediction, AbsTensor<T> label,
                   std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Sink, "MSE"), m_id++,
                                std::move(name) };

    const auto prevUnitId = prediction.GetPrevOutput();
    const auto predictionShape = prediction.GetShape();
    const auto labelShape = label.GetShape();

    UnitMetaData<T> unitMetaData(subjectUnitId, m_batchSize, {}, {},
                                 { { "prediction", predictionShape },
                                   { "label", labelShape } }, Shape(),
                                 { { "prediction", prevUnitId },
                                   { "label", labelShape } }, m_device);
}

template <typename T>
void Model<T>::Compile(std::string optimizer, Parameter optimizerParams)
{
    m_unitManager.Compile(optimizer, optimizerParams);
}

template <typename T>
void Model<T>::Fit(std::size_t epochs)
{
    for (std::size_t cycle = 0; cycle < epochs; ++cycle)
    {
        m_unitManager.Forward(cycle);
        m_unitManager.Backward(cycle);
    }
}
}

#endif

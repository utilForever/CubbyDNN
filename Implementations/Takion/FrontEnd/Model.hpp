// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_FRONTEND_MODEL_HPP
#define TAKION_FRONTEND_MODEL_HPP

#include <Takion/FrontEnd/ModelDecl.hpp>
#include <Takion/FrontEnd/AbsTensorDecl.hpp>
#include <Takion/Units/UnitType.hpp>
#include <Takion/Computations/Device.hpp>
#include <Takion/Engine/UnitManager.hpp>


namespace Takion::FrontEnd
{
template <typename T>
Model<T>::Model(Compute::Device device, std::size_t batchSize)
    : m_device(std::move(device)),
      m_unitManager(Engine::UnitManager<T>(batchSize)),
      m_batchSize(batchSize)
{
}

template <typename T>
void Model<T>::SetDevice(Compute::Device device)
{
    m_device = device;
}

template <typename T>
AbsTensor<T> Model<T>::PlaceHolder(const Shape& shape,
                                   std::function<std::vector<T>()>
                                   loaderFunction, std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Source, "PlaceHolder"),
                                m_id++, std::move(name) };

    UnitMetaData<T> unitMetaData(subjectUnitId, m_batchSize, {}, {}, {}, shape,
                                 {}, m_device);

    m_unitManager.AppendUnit(std::move(unitMetaData));
    m_unitManager.SetLoader(subjectUnitId, loaderFunction);

    return AbsTensor<T>(shape, subjectUnitId);
}

template <typename T>
AbsTensor<T> Model<T>::Constant(const Shape& shape, std::vector<T> data,
                                std::string name)
{
    if (data.size() != shape.Size() * m_batchSize)
    {
        const std::string errorMessage =
            std::string("Constant - ") + name +
            " Given data and shape's size should be identical" +
            " shape : " + shape.ToString() +
            " dataSize : " + std::to_string(data.size());
        throw std::invalid_argument(errorMessage);
    }

    const UnitId subjectUnitId{ UnitType(UnitBaseType::Source, "Constant"),
                                m_id++, std::move(name) };

    std::unordered_map<std::string, std::unique_ptr<Compute::Initializer<T>>>
        initializerMap;
    initializerMap["vectorInitializer"] =
        std::make_unique<Compute::VectorInitializer<T>>(data);

    UnitMetaData<T> unitMetaData(subjectUnitId, m_batchSize, {},
                                 std::move(initializerMap),
                                 {}, shape,
                                 {}, m_device);
    m_unitManager.AppendUnit(std::move(unitMetaData));

    return AbsTensor<T>(shape, subjectUnitId);
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

    m_appendSubjectUnitToPreviousOutput(subjectUnitId, prevUnitId);

    const Shape weightShape({ prevOutputShape.NumCol(), numUnits });
    const Shape biasShape({ numUnits });
    const Shape outputShape({ numUnits });

    std::unordered_map<std::string, std::unique_ptr<Compute::Initializer<T>>>
        initializerMap;

    weightInitializer->FanIn = prevOutputShape.NumCol();
    weightInitializer->FanOut = numUnits;
    biasInitializer->FanIn = prevOutputShape.NumCol();
    biasInitializer->FanOut = numUnits;

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
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "ReLU"), m_id++,
                                std::move(name) };

    const auto prevUnitId = source.GetPrevOutput();
    const auto shape = source.GetShape();

    m_appendSubjectUnitToPreviousOutput(subjectUnitId, prevUnitId);

    std::unordered_map<std::string, std::unique_ptr<Compute::Initializer<T>>>
        initializerMap;

    UnitMetaData<T> unitMetaData(
        subjectUnitId, m_batchSize, {}, {}, { { "input", shape } },
        shape, { { "input", prevUnitId } }, m_device);

    m_unitManager.AppendUnit(std::move(unitMetaData));

    return AbsTensor<T>(shape, subjectUnitId);
}

template <typename T>
AbsTensor<T> Model<T>::Sigmoid(AbsTensor<T> source, std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "Sigmoid"),
                                m_id++,
                                std::move(name) };

    const auto prevUnitId = source.GetPrevOutput();
    const auto shape = source.GetShape();

    m_appendSubjectUnitToPreviousOutput(subjectUnitId, prevUnitId);

    UnitMetaData<T> unitMetaData(subjectUnitId, m_batchSize, {}, {},
                                 { { "input", shape } }, shape,
                                 { { "input", prevUnitId } }, m_device);

    m_unitManager.AppendUnit(std::move(unitMetaData));

    return AbsTensor<T>(shape, subjectUnitId);
}

template <typename T>
AbsTensor<T> Model<T>::SoftMax(AbsTensor<T> source, std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "SoftMax"),
                                m_id++, std::move(name) };

    const auto prevUnitId = source.GetPrevOutput();
    const auto shape = source.GetShape();

    m_appendSubjectUnitToPreviousOutput(subjectUnitId, prevUnitId);

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

    const auto predictionId = prediction.GetPrevOutput();
    const auto labelId = label.GetPrevOutput();
    const auto predictionShape = prediction.GetShape();
    const auto labelShape = label.GetShape();

    m_appendSubjectUnitToPreviousOutput(subjectUnitId, predictionId);
    m_appendSubjectUnitToPreviousOutput(subjectUnitId, labelId);

    UnitMetaData<T> unitMetaData(subjectUnitId, m_batchSize, {}, {},
                                 { { "prediction", predictionShape },
                                   { "label", labelShape } }, Shape(),
                                 { { "prediction", predictionId },
                                   { "label", labelId } }, m_device);

    m_unitManager.AppendUnit(std::move(unitMetaData));
}

template <typename T>
void Model<T>::CrossEntropy(AbsTensor<T> prediction, AbsTensor<T> label,
                            std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Sink, "CrossEntropy"),
                                m_id++,
                                std::move(name) };

    const auto predictionId = prediction.GetPrevOutput();
    const auto labelId = label.GetPrevOutput();
    const auto predictionShape = prediction.GetShape();
    const auto labelShape = label.GetShape();

    m_appendSubjectUnitToPreviousOutput(subjectUnitId, predictionId);
    m_appendSubjectUnitToPreviousOutput(subjectUnitId, labelId);

    UnitMetaData<T> unitMetaData(
        subjectUnitId, m_batchSize, {}, {},
        { { "prediction", predictionShape }, { "label", labelShape } }, Shape(),
        { { "prediction", predictionId }, { "label", labelId } }, m_device);

    m_unitManager.AppendUnit(std::move(unitMetaData));
}


template <typename T>
void Model<T>::Compile(std::string optimizer, Parameter optimizerParams)
{
    m_unitManager.Compile(optimizer, optimizerParams);
}

template <typename T>
void Model<T>::Train(std::size_t cycle)
{
    m_unitManager.Forward(cycle);
    m_unitManager.Backward(cycle);
}

template <typename T>
void Model<T>::Predict()
{
    m_unitManager.Forward(0);
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

template <typename T>
std::tuple<std::vector<T>, Shape, std::size_t> Model<T>::Output(
    AbsTensor<T> absTensor) const
{
    auto unitId = absTensor.GetPrevOutput();
    const auto& tensor = m_unitManager.GetOutput(unitId);
    const auto size = tensor.TensorShape.Size();

    std::vector<T> data(tensor.BatchSize * size);

    for (long batchIdx = 0; batchIdx < tensor.BatchSize; ++batchIdx)
        for (std::size_t i = 0; i < size; ++i)
        {
            const auto idx = batchIdx * size + i;
            data[idx] = tensor.At(idx);
        }

    return std::make_tuple(data, tensor.TensorShape, tensor.BatchSize);
}


template <typename T>
void Model<T>::m_appendSubjectUnitToPreviousOutput(
    const UnitId& subjectUnit, const UnitId& previousUnit)
{
    m_unitManager.GetUnitMetaData(previousUnit).AppendOutputUnitId(
        subjectUnit);
}
}

#endif

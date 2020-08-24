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
#include <Takion/Utils/Loaders/Loader.hpp>
#include <memory>


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
AbsTensor<T> Model<T>::Fetcher(const Shape& shape,
                               std::unique_ptr<Util::Loader<T>> loaderFunction,
                               std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Fetcher, "Fetcher"),
                                m_id++, std::move(name) };

    UnitMetaData<T> unitMetaData(subjectUnitId, m_batchSize, {}, {}, {}, shape,
                                 {}, m_device);

    m_unitManager.AppendUnit(std::move(unitMetaData));
    m_unitManager.SetLoader(subjectUnitId, std::move(loaderFunction));

    return AbsTensor<T>(shape, subjectUnitId);
}

template <typename T>
AbsTensor<T> Model<T>::Fetcher(const Shape& shape,
                               std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Fetcher, "Fetcher"),
                                m_id++, std::move(name) };

    UnitMetaData<T> unitMetaData(subjectUnitId, m_batchSize, {}, {}, {}, shape,
                                 {}, m_device);

    m_unitManager.AppendUnit(std::move(unitMetaData));
    m_unitManager.SetLoader(subjectUnitId,
                            std::make_unique<Util::Loader<T>>(
                                shape, m_batchSize));

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

    const UnitId subjectUnitId{ UnitType(UnitBaseType::Constant, "Constant"),
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
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Activation, "ReLU"),
                                m_id++,
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
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Activation, "Sigmoid"),
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
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Activation, "SoftMax"),
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
AbsTensor<T> Model<T>::MSE(AbsTensor<T> prediction, AbsTensor<T> label,
                           std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Loss, "MSE"), m_id++,
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
    return AbsTensor<T>(Shape(), subjectUnitId);
}

template <typename T>
AbsTensor<T> Model<T>::CrossEntropy(AbsTensor<T> prediction, AbsTensor<T> label,
                                    std::string name)
{
    const UnitId subjectUnitId{ UnitType(UnitBaseType::Loss, "CrossEntropy"),
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
    return AbsTensor<T>(Shape(), subjectUnitId);
}


template <typename T>
void Model<T>::Compile(std::string optimizer, Parameter optimizerParams)
{
    m_unitManager.Compile(optimizer, optimizerParams);
}

template <typename T>
void Model<T>::Train()
{
    m_unitManager.Forward();
    m_unitManager.Backward();
    m_unitManager.ResetState();
}


template <typename T>
void Model<T>::Train(std::map<AbsTensor<T>, std::vector<T>>
                     inputDataMap,
                     AbsTensor<T> labelUnit,
                     std::vector<T> label)
{
    for (const auto& [inputUnit, trainData] : inputDataMap)
    {
        const auto inputUnitId = inputUnit.GetPrevOutput();
        auto& dataFetcher = dynamic_cast<Graph::PlaceHolder<T>*>(
                m_unitManager.GetUnit(inputUnitId).get())
            ->GetLoader();
        dataFetcher->SetData(trainData);
    }

    const auto labelUnitId = labelUnit.GetPrevOutput();
    auto& labelFetcher = dynamic_cast<Graph::PlaceHolder<T>*>(
            m_unitManager.GetUnit(labelUnitId).get())
        ->GetLoader();
    labelFetcher->SetData(label);
    m_unitManager.Forward();
    m_unitManager.Backward();
    m_unitManager.ResetState();
}


template <typename T>
void Model<T>::Predict()
{
    m_unitManager.Forward();
}

template <typename T>
void Model<T>::Predict(std::map<AbsTensor<T>, std::vector<T>> inputDataMap)
{
    for (const auto& [inputUnit, trainData] : inputDataMap)
    {
        const auto inputUnitId = inputUnit.GetPrevOutput();
        auto& dataFetcher = dynamic_cast<Graph::PlaceHolder<T>*>(
                m_unitManager.GetUnit(inputUnitId).get())
            ->GetLoader();
        dataFetcher->SetData(trainData);
    }
    m_unitManager.Forward();
    m_unitManager.ResetState();
}


template <typename T>
void Model<T>::Predict(
    std::map<AbsTensor<T>, std::vector<T>> inputDataMap, AbsTensor<T> labelUnit,
    std::vector<T> label)
{
    for (const auto& [inputUnit, trainData] : inputDataMap)
    {
        const auto inputUnitId = inputUnit.GetPrevOutput();
        auto& dataFetcher = dynamic_cast<Graph::PlaceHolder<T>*>(
                m_unitManager.GetUnit(inputUnitId).get())
            ->GetLoader();
        dataFetcher->SetData(trainData);
    }

    const auto labelUnitId = labelUnit.GetPrevOutput();
    auto& labelFetcher = dynamic_cast<Graph::PlaceHolder<T>*>(
            m_unitManager.GetUnit(labelUnitId).get())
        ->GetLoader();
    labelFetcher->SetData(label);

    m_unitManager.Forward();
    m_unitManager.ResetState();
}


template <typename T>
void Model<T>::Fit(std::size_t epochs)
{
    for (std::size_t cycle = 0; cycle < epochs; ++cycle)
    {
        Train();
    }
}

template <typename T>
Util::TensorData<T> Model<T>::Output(
    AbsTensor<T> absTensor) const
{
    auto unitId = absTensor.GetPrevOutput();
    const auto& tensor = m_unitManager.GetOutput(unitId);
    const auto size = tensor.TensorShape.Size();

    std::vector<T> data(tensor.BatchSize * size);

    for (long batchIdx = 0; batchIdx < static_cast<long>(tensor.BatchSize); ++batchIdx)
        for (std::size_t i = 0; i < size; ++i)
        {
            const auto idx = batchIdx * size + i;
            data[idx] = tensor.At(idx);
        }

    return Util::TensorData<T>(data, tensor.TensorShape, tensor.BatchSize);
}

template <typename T>
T Model<T>::GetLoss(AbsTensor<T> lossId)
{
    const auto unitId = lossId.GetPrevOutput();
    if (unitId.Type.BaseType != UnitBaseType::Loss)
        throw std::invalid_argument("Given unit must be loss");

    T loss = m_unitManager.GetUnit(unitId)->GetLoss();
    return loss;
}


template <typename T>
void Model<T>::ChangeLoader(AbsTensor<T> loaderId,
                            std::function<std::vector<T>()> loaderFunction)
{
    const auto unitId = loaderId.GetPrevOutput();
    if (unitId.Type.BaseType != UnitBaseType::Fetcher)
        throw std::invalid_argument("Given unit must be Fetcher");

    dynamic_cast<Graph::PlaceHolder<T>*>(m_unitManager.GetUnit(unitId).get())
        ->SetLoader(loaderFunction);
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

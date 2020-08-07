// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef Takion_MODEL_IMPL_HPP
#define Takion_MODEL_IMPL_HPP

#include <Takion/Computations/Activations/ActivationWrapper.hpp>
#include <Takion/Computations/LossFunctions/LossFunctionWrapper.hpp>
#include <Takion/Engine/ModelDecl.hpp>
#include <Takion/Units/UnitType.hpp>

namespace Takion::Graph
{
template <typename T>
Model<T>::Model()
{
}

template <typename T>
UnitId Model<T>::DataLoader(const Shape& shape, const std::string& name,
                            Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Source, "DataLoader"), m_id++,
                          name };
    UnitMetaData unitMetaData(subjectUnitId, {}, {}, {}, shape, {},
                              std::move(device));
    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

template <typename T>
UnitId Model<T>::Dense(const UnitId& input, std::size_t units,
                       std::unique_ptr<Initializer<T>> weightInitializer,
                       std::unique_ptr<Initializer<T>> biasInitializer,
                       const std::string& name, Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "Dense"), m_id++,
                          name };
    const auto previousOutputShape = m_unitManager.GetUnitOutputShape(input);

    Shape weightShape({ units, previousOutputShape.NumRow() });
    Shape biasShape({ units, 1 });
    Shape outputShape({ units, previousOutputShape.NumCol() });

    std::unordered_map<std::string, std::unique_ptr<Initializer<T>>>
        initializerMap;

    initializerMap["weight"] = std::move(weightInitializer);
    initializerMap["bias"] = std::move(biasInitializer);

    UnitMetaData unitMetaData(
        subjectUnitId, { { "weight", weightShape }, { "bias", biasShape } },
        std::move(initializerMap), { { "input", previousOutputShape } },
        outputShape, { { "input", input } }, std::move(device));

    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

template <typename T>
UnitId Model<T>::Activation(const UnitId& input,
                            const std::string& activationName,
                            const std::string& name, Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Hidden, "Activation"), m_id++,
                          name };
    const auto previousOutputShape = m_unitManager.GetUnitOutputShape(input);

    UnitMetaData unitMetaData(
        subjectUnitId, {}, {}, { { "input", previousOutputShape } },
        { previousOutputShape }, { { "input", input } },
        std::move(device), Parameter({ { "activationName", activationName } }));

    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

template <typename T>
UnitId Model<T>::Constant(Tensor tensor, const std::string& name)
{

    UnitId subjectUnitId{ UnitType(UnitBaseType::Source, "Constant"), m_id++,
                          name };
    UnitMetaData unitMetaData(subjectUnitId, {}, {}, {}, tensor.TensorShape, {},tensor.Device);
    unitMetaData.AddInternalTensor("constant", std::move(tensor));
    m_unitManager.AppendUnit(std::move(unitMetaData));
    return subjectUnitId;
}

template <typename T>
void Model<T>::Compile(const std::string& optimizer,
                       Parameter optimizerParams) noexcept
{
    Compute::ActivationWrapper<T>::Initialize();
    Compute::LossFunctionWrapper<T>::Initialize();
    m_unitManager.Compile(optimizer, optimizerParams);
}

template <typename T>
UnitId Model<T>::Loss(const UnitId& prediction, const UnitId& label,
                      std::string lossType, const std::string& name,
                      Compute::Device device)
{
    UnitId subjectUnitId{ UnitType(UnitBaseType::Sink, "Loss"), m_id++, name };
    const auto predictionShape = m_unitManager.GetUnitOutputShape(prediction);
    const auto labelShape = m_unitManager.GetUnitOutputShape(label);

    UnitMetaData unitMetaData(
        subjectUnitId, {}, {},
        { { "prediction", predictionShape }, { "label", labelShape } }, Shape(),
        { { "prediction", prediction }, { "label", label } },
        std::move(device), Parameter({ { "lossType", lossType } }));
    m_unitManager.AppendUnit(std::move(unitMetaData));

    return subjectUnitId;
}

template <typename T>
void Model<T>::Train(std::size_t epochs, bool async)
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
} // namespace Takion::Graph

#endif

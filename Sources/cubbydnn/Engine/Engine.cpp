// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbyDnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Engine/Engine.hpp>

namespace CubbyDNN
{
Graph::Graph(NumberSystem numberSystem)
    : m_numberSystem(numberSystem)
{
}

void Graph::ExecuteForward(std::size_t epochs)
{
    m_maxEpochs = epochs;
    const auto totalDepth = m_executionOrder.size();
    for (std::size_t count = 0; count < epochs; ++count)
    {
        //! Forard Propagation
        for (std::size_t depth = 0; depth < totalDepth; ++depth)
        {
            auto unitVector = m_executionOrder.at(depth);
            for (const auto& unit : unitVector)
            {
                if (unit.Type == UnitType::Source)
                {
                    m_sourceUnitVector.at(unit.ID)->Forward();
                    m_sourceCopyUnitVector.at(unit.ID)->Forward();
                }
                else if (unit.Type == UnitType::Hidden)
                {
                    m_hiddenUnitVector.at(unit.ID)->Forward();
                    m_hiddenCopyUnitVector.at(unit.ID)->Forward();
                }
                else if (unit.Type == UnitType::Sink)
                {
                    m_sinkUnit->Forward();
                }
            }
        }
    }
}

void Graph::Fit(std::size_t epochs)
{
    const auto totalDepth = m_executionOrder.size();
    for (std::size_t count = 0; count < epochs; ++count)
    {
        //! Forard Propagation
        for (std::size_t depth = 0; depth < totalDepth; ++depth)
        {
            auto unitVector = m_executionOrder.at(depth);
            for (const auto& unit : unitVector)
            {
                if (unit.Type == UnitType::Source)
                {
                    m_sourceUnitVector.at(unit.ID)->Forward();
                    m_sourceCopyUnitVector.at(unit.ID)->Forward();
                }
                else if (unit.Type == UnitType::Hidden)
                {
                    m_hiddenUnitVector.at(unit.ID)->Forward();
                    m_hiddenCopyUnitVector.at(unit.ID)->Forward();
                }
                else if (unit.Type == UnitType::Sink)
                {
                    m_sinkUnit->Forward();
                }
            }
        }

        //! Back propagation
        for (std::size_t depth = totalDepth; depth >= 0; --depth)
        {
            auto unitVector = m_executionOrder.at(depth);
            for (const auto& unit : unitVector)
            {
                if (unit.Type == UnitType::Source)
                    m_sourceUnitVector.at(unit.ID)->Backward();
                else if (unit.Type == UnitType::Hidden)
                    m_hiddenUnitVector.at(unit.ID)->Backward();
                else if (unit.Type == UnitType::Sink)
                    m_sinkUnit->Backward();
            }
            if (depth > 0)
            {
                auto previousUnitVector = m_executionOrder.at(depth - 1);
                for (const auto& unit : previousUnitVector)
                {
                    if (unit.Type == UnitType::Source)
                        m_sourceCopyUnitVector.at(unit.ID)->Backward();
                    else if (unit.Type == UnitType::Hidden)
                        m_hiddenCopyUnitVector.at(unit.ID)->Backward();
                }
            }
        }
    }
}

UnitId Graph::PlaceHolder(const Shape& shape)
{
    SharedPtr<PlaceHolderUnit> ptr =
        SharedPtr<PlaceHolderUnit>::Make(TensorInfo(shape, m_numberSystem));

    const auto id = m_sourceUnitVector.size();
    m_sourceUnitVector.emplace_back(std::move(ptr));
    return { UnitType::Source, id };
}

// TODO : put activation, initializing methods, etc.
UnitId Graph::Dense(const UnitId& input, std::size_t units, ActivationType activationType)
{
    TensorInfo inputTensorInfo;
    TensorInfo weightTensorInfo;
    TensorInfo biasTensorInfo;
    if (input.Type == UnitType::Hidden)
    {
        inputTensorInfo =
            m_hiddenUnitVector.at(input.ID)->GetOutputTensorInfo();
    }
    else if (input.Type == UnitType::Source)
    {
        inputTensorInfo =
            m_hiddenUnitVector.at(input.ID)->GetOutputTensorInfo();
    }
    else
        throw std::runtime_error("input unit must be source or hidden");

    const Shape weightShape = { inputTensorInfo.GetShape().Row(), units };
    const Shape biasShape = { 1, inputTensorInfo.GetShape().Row() };

    // TODO : specify number system other than float
    TensorInfo weightInfo(weightShape);
    TensorInfo biasInfo(biasShape);

    SharedPtr<ConstantUnit> weightPtr = SharedPtr<ConstantUnit>::Make(
        weightInfo, AllocateData<float>(weightShape));
    SharedPtr<ConstantUnit> biasPtr = SharedPtr<ConstantUnit>::Make(
        weightInfo, AllocateData<float>(biasShape));

    const auto weightUnitID = m_sourceUnitVector.size();
    m_sourceUnitVector.emplace_back(std::move(weightPtr));
    const auto biasUnitID = m_sourceUnitVector.size();
    m_sourceUnitVector.emplace_back(std::move(biasPtr));

    const UnitId weightUnit = { UnitType::Source, weightUnitID };
    const UnitId biasUnit = { UnitType::Source, biasUnitID };

    const auto denseUnitID = m_hiddenUnitVector.size();
    SharedPtr<DenseUnit> densePtr = SharedPtr<DenseUnit>::Make(
        inputTensorInfo, weightTensorInfo, biasTensorInfo);

    densePtr->SetInputUnitVector({ input, weightUnit, biasUnit });

    m_hiddenUnitVector.emplace_back(SharedPtr<DenseUnit>::Make(
        inputTensorInfo, weightTensorInfo, biasTensorInfo));

    const UnitId unitIdentifier = { UnitType::Hidden, denseUnitID };
    return unitIdentifier;
}

//! TODO : Do dfs search to seek and connect units together
void Graph::Compile(Optimizer optimizer, Loss loss)
{
    auto identifier = m_sinkUnit->GetId();
    std::vector<UnitId> sinkUnitVector;
    sinkUnitVector.emplace_back(identifier);
    auto inputUnitVector = m_sinkUnit->GetInputUnitVector();
    m_executionOrder.emplace_back(sinkUnitVector);
    for (auto unit : inputUnitVector)
    {
        m_getExecutionOrder(unit, m_executionOrder, 1);
    }

    for (auto unitIdVector : m_executionOrder)
    {
        for (auto unitId : unitIdVector)
        {
            SharedPtr<CopyUnit> copyUnit = SharedPtr<CopyUnit>::Make();

            if (unitId.Type == UnitType::Source)
            {
                auto& unitPtr = m_sourceUnitVector.at(unitId.ID);
                copyUnit->SetInputPtr(unitPtr);
                const auto outputUnitIdVector = unitPtr->GetOutputUnitVector();
                std::vector<SharedPtr<ComputableUnit>> outputUnitVector;
                outputUnitVector.reserve(outputUnitIdVector.size());
                for (const auto& outputUnitIdPair : outputUnitIdVector)
                {
                    auto [outputUnitId, inputIndex] = outputUnitIdPair;

                    if (outputUnitId.Type == UnitType::Hidden)
                    {
                        copyUnit->AddOutputPtr(
                            m_hiddenUnitVector.at(outputUnitId.ID), inputIndex);
                    }
                    else if (outputUnitId.Type == UnitType::Sink)
                    {
                        copyUnit->AddOutputPtr(m_sinkUnit, inputIndex);
                    }
                }
            }
            else if (unitId.Type == UnitType::Hidden)
            {
                auto& unitPtr = m_sourceUnitVector.at(unitId.ID);
                copyUnit->SetInputPtr(unitPtr);
                const auto outputUnitIdVector = unitPtr->GetOutputUnitVector();
                std::vector<SharedPtr<ComputableUnit>> outputUnitVector;
                outputUnitVector.reserve(outputUnitIdVector.size());
                for (const auto& outputUnitIdPair : outputUnitIdVector)
                {
                    auto [outputUnitId, inputIndex] = outputUnitIdPair;

                    if (outputUnitId.Type == UnitType::Hidden)
                    {
                        copyUnit->AddOutputPtr(
                            m_hiddenUnitVector.at(outputUnitId.ID), inputIndex);
                    }
                    else if (outputUnitId.Type == UnitType::Sink)
                    {
                        copyUnit->AddOutputPtr(m_sinkUnit, inputIndex);
                    }
                }
            }
        }
    }
}

void Graph::m_getExecutionOrder(
    UnitId subjectUnit, std::vector<std::vector<UnitId>>& executionOrder,
    int depth)
{
    const UnitType type = subjectUnit.Type;
    const std::size_t id = subjectUnit.ID;
    if (executionOrder.size() < depth + 1)
        executionOrder.emplace_back(std::vector<UnitId>());

    if (type == UnitType::Hidden)
    {
        const auto& hiddenUnit = m_hiddenUnitVector.at(id);
        const auto& inputUnitIdVector = hiddenUnit->GetInputUnitVector();
        executionOrder.at(depth).emplace_back(hiddenUnit->GetId());
        for (const auto& unit : inputUnitIdVector)
        {
            m_getExecutionOrder(unit, executionOrder, depth + 1);
        }
    }
    else if (type == UnitType::Source)
    {
        const auto& sourceUnit = m_sourceUnitVector.at(id);
        executionOrder.at(depth).emplace_back(sourceUnit->GetId());
    }
}
} // namespace CubbyDNN

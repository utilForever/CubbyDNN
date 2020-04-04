// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Engine/Graph.hpp>

namespace CubbyDNN
{
Graph::Graph(NumberSystem numberSystem)
    : m_numberSystem(numberSystem)
{
}

void Graph::Predict(std::size_t epochs)
{
    m_maxEpochs = epochs;
    const auto totalDepth = m_executionOrder.size();
    for (std::size_t count = 0; count < epochs; ++count)
    {
        //! Forward Propagation
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
        //! Forward Propagation
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
    const auto id = m_sourceUnitVector.size();
    UnitId unitId = { UnitType::Source, id };
    SharedPtr<PlaceHolderUnit> ptr =
        SharedPtr<PlaceHolderUnit>::Make(unitId, shape, m_numberSystem);

    m_sourceUnitVector.emplace_back(std::move(ptr));
    return unitId;
}

// TODO : put activation, initializing methods, etc.
UnitId Graph::Dense(const UnitId& input, std::size_t units,
                    Activation activation, InitializerType kernelInitializer,
                    InitializerType biasInitializer, float dropoutRate)
{
    Shape inputShape;
    if (input.Type == UnitType::Hidden)
    {
        inputShape = m_hiddenUnitVector.at(input.ID)->GetOutputTensorShape();
    }
    else if (input.Type == UnitType::Source)
    {
        inputShape = m_hiddenUnitVector.at(input.ID)->GetOutputTensorShape();
    }
    else
        throw std::runtime_error("input unit must be source or hidden");

    const Shape weightShape = { inputShape.NumRows(), units };
    const Shape biasShape = { 1, inputShape.NumRows() };
    Shape outputShape = inputShape;
    outputShape.SetNumCols(units);

    const auto denseUnitID = m_hiddenUnitVector.size();
    const UnitId unitId = { UnitType::Hidden, denseUnitID };
    SharedPtr<DenseUnit> densePtr = SharedPtr<DenseUnit>::Make(
        unitId, inputShape, weightShape, biasShape, outputShape, m_numberSystem,
        kernelInitializer, biasInitializer, activation, dropoutRate);

    densePtr->SetInputUnitVector({ input });

    m_hiddenUnitVector.emplace_back(densePtr);
    return unitId;
}

void Graph::Compile(UnitId unitId, OptimizerType optimizer, Loss loss)
{
    m_optimizer = optimizer;
    UnitId sinkUnitId = { UnitType::Sink, 0 };
    if (loss == Loss::CrossEntropy)
    {
        m_sinkUnit = SharedPtr<CrossEntropy>::Make(
            sinkUnitId,
            m_hiddenUnitVector.at(unitId.ID)->GetOutputTensorShape(),
            m_numberSystem);
    }
    else
    {
        // TODO : implement other kinds of loss functions
        throw std::runtime_error("Not Implemented");
    }

    auto identifier = m_sinkUnit->GetId();
    std::vector<UnitId> sinkUnitVector;
    sinkUnitVector.emplace_back(identifier);
    auto inputUnitVector = m_sinkUnit->GetInputUnitVector();
    m_executionOrder.emplace_back(sinkUnitVector);
    for (auto unit : inputUnitVector)
    {
        m_getExecutionOrder(unit, m_executionOrder, 1);
    }

    for (const auto& unitIdVector : m_executionOrder)
    {
        for (auto id : unitIdVector)
        {
            SharedPtr<CopyUnit> copyUnit = SharedPtr<CopyUnit>::Make();

            if (id.Type == UnitType::Source)
            {
                auto& unitPtr = m_sourceUnitVector.at(id.ID);
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
            else if (id.Type == UnitType::Hidden)
            {
                auto& unitPtr = m_sourceUnitVector.at(id.ID);
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
    if (executionOrder.size() < static_cast<std::size_t>(depth + 1))
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

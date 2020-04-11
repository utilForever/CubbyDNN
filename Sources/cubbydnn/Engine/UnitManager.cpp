// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#include <cubbydnn/Engine/UnitManager.hpp>

namespace CubbyDNN::Graph
{
UnitManager::UnitManager(std::size_t totalUnitSize)
{
    m_sourceUnitVector.reserve(totalUnitSize);
}

UnitManager::UnitManager(UnitManager&& unitManager) noexcept
    : m_sourceUnitVector(std::move(unitManager.m_sourceUnitVector)),
      m_hiddenUnitVector(std::move(unitManager.m_hiddenUnitVector)),
      m_sinkUnit(std::move(unitManager.m_sinkUnit)),
      m_copyUnitMap(std::move(unitManager.m_copyUnitMap))
{
}

UnitManager& UnitManager::operator=(UnitManager&& unitManager) noexcept
{
    if (this == &unitManager)
        return *this;
    m_sourceUnitVector = std::move(unitManager.m_sourceUnitVector);
    m_hiddenUnitVector = std::move(unitManager.m_hiddenUnitVector);
    m_sinkUnit = std::move(unitManager.m_sinkUnit);
    m_copyUnitMap = std::move(unitManager.m_copyUnitMap);
    return *this;
}

void UnitManager::AddUnit(UnitId previousUnitId, std::size_t parameterIndex,
                          SharedPtr<ComputableUnit> unit)
{
    const auto unitId = unit->GetId();
    if (unitId.BaseType != UnitBaseType::Source)
    {
        auto prevUnit = GetUnit(previousUnitId);
        prevUnit->AddOutputUnitVector(unit->GetId(), parameterIndex);
    }

    if (unitId.BaseType == UnitBaseType::Source)
        m_sourceUnitVector.emplace_back(std::move(unit));
    else if (unitId.BaseType == UnitBaseType::Hidden)
        m_hiddenUnitVector.emplace_back(std::move(unit));
    else if (unitId.BaseType == UnitBaseType::Sink)
        m_sinkUnit = std::move(unit);
    else
        throw std::invalid_argument("Unsupported BaseUnitType");
}

SharedPtr<ComputableUnit> UnitManager::GetUnit(UnitId unitId)
{
    if (unitId.BaseType == UnitBaseType::Source)
        return m_sourceUnitVector.at(unitId.Id);
    if (unitId.BaseType == UnitBaseType::Hidden)
        return m_hiddenUnitVector.at(unitId.Id);
    if (unitId.BaseType == UnitBaseType::Sink)
        return m_sinkUnit;
    throw std::invalid_argument("Unsupported BaseUnitType");
}

SharedPtr<CopyUnit> UnitManager::GetCopyUnit(UnitId unitId)
{
    return m_copyUnitMap[unitId];
}


void UnitManager::CreateExecutionOrder()
{
    if (!m_executionOrder.empty())
        m_executionOrder.clear();
    m_createExecutionOrder(m_sinkUnit->GetId(), 0);
}

void UnitManager::AssignCopyUnits()
{
    for (const auto& executionGroup : m_executionOrder)
    {
        for (const auto& subjectUnitId : executionGroup)
        {
            if (subjectUnitId.BaseType == UnitBaseType::Sink)
                continue;

            auto getOutputUnitVector = GetUnit(subjectUnitId)->
                GetOutputUnitVector();
            auto copyUnit = SharedPtr<CopyUnit>::Make();
            for (const auto& unitIdIndexPair : getOutputUnitVector)
            {
                auto [destUnitId, parameterIndex] = unitIdIndexPair;
                copyUnit->AddOutputPtr(GetUnit(destUnitId), parameterIndex);
                copyUnit->SetInputPtr(GetUnit(subjectUnitId));
            }
            m_copyUnitMap.insert_or_assign(subjectUnitId, std::move(copyUnit));
        }
    }
}


void UnitManager::m_createExecutionOrder(UnitId unitId, std::size_t depth)
{
    if (m_executionOrder.size() <= depth)
        m_executionOrder.emplace_back(std::vector<UnitId>());
    m_executionOrder.at(depth).emplace_back(unitId);

    auto inputUnitVector = GetUnit(unitId)->GetInputUnitVector();
    for (const auto& inputUnitId : inputUnitVector)
    {
        m_createExecutionOrder(inputUnitId, depth++);
    }
}

void UnitManager::Train(std::size_t epochs)
{
    const auto sequenceSize = m_executionOrder.size();
    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
        for (int index = static_cast<int>(sequenceSize); index >= 0; --index)
        {
            for (auto unitId : m_executionOrder.at(index))
            {
                auto unit = GetUnit(unitId);
                if (unit->GetForwardStateCount() == epoch)
                {
                    unit->Forward();
                    if (unit->GetId().BaseType != UnitBaseType::Sink)
                        GetCopyUnit(unitId)->Forward();
                    unit->IncrementForwardStateCount();
                }
            }
        }

        for (int index = 0; index < static_cast<int>(sequenceSize); ++index)
        {
            for (auto unitId : m_executionOrder.at(index))
            {
                auto unit = GetUnit(unitId);
                if (unit->GetBackwardStateCount() == epoch)
                {
                    if (unit->GetId().BaseType != UnitBaseType::Sink)
                        GetCopyUnit(unitId)->Backward();
                    unit->Backward();
                    unit->IncrementBackwardStateCount();
                }
            }
        }
    }
}

void UnitManager::Predict()
{
    const auto sequenceSize = m_executionOrder.size();
    for (int index = static_cast<int>(sequenceSize); index >= 0; --index)
    {
        for (auto unitId : m_executionOrder.at(index))
        {
            auto unit = GetUnit(unitId);
            unit->Forward();
            if (unit->GetId().BaseType != UnitBaseType::Sink)
                GetCopyUnit(unitId)->Forward();
            unit->IncrementForwardStateCount();
        }
    }
}
}

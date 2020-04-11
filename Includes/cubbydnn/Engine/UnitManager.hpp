// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_UNITMANAGER_HPP
#define CUBBYDNN_UNITMANAGER_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Units/CopyUnit.hpp>
#include <cubbydnn/Units/SinkComputableUnits/SinkUnit.hpp>
#include <vector>
#include <unordered_map>

namespace CubbyDNN::Graph
{
class UnitManager
{
public:
    UnitManager(std::size_t totalUnitSize);
    virtual ~UnitManager() = default;

    UnitManager(const UnitManager& unitManager) = delete;
    UnitManager(UnitManager&& unitManager) noexcept;
    UnitManager& operator=(const UnitManager& unitManager) = delete;
    UnitManager& operator=(UnitManager&& unitManager) noexcept;

    //! Adds new unit to unitManager
    void AddUnit(UnitId previousUnitId, std::size_t parameterIndex,
                 SharedPtr<ComputableUnit> unit);

    //! Gets unit that matches unitId
    SharedPtr<ComputableUnit> GetUnit(UnitId unitId);

    //! Gets CopyUnit that connects output of given unitId
    SharedPtr<CopyUnit> GetCopyUnit(UnitId unitId);

    //! Creates Execution order
    void CreateExecutionOrder();

    void AssignCopyUnits();

    //! Creates Unit with given unit type and information
    template <typename T, typename... Ts>
    UnitId CreateSource(
        const std::string& unitName, Ts&&... args)
    {
        const std::size_t id = m_sourceUnitVector.size();
        UnitId unitId{ UnitBaseType::Source, id, unitName };
        this->AddUnit(UnitId(),
                      SharedPtr<T>(unitId, std::forward(args...)));
        return unitId;
    }

    //! Creates Unit with given unit type and information
    template <typename T, typename... Ts>
    UnitId CreateHidden(UnitId previousUnitId,
                        const std::string& unitName,
                        Ts&&... args)
    {
        const std::size_t id = m_hiddenUnitVector.size();
        UnitId unitId{ UnitBaseType::Hidden, id, unitName };
        this->AddUnit(previousUnitId,
                      SharedPtr<T>(unitId, std::forward(args...)));
        return unitId;
    }

    template <typename T, typename... Ts>
    UnitId CreateSink(UnitId previousUnitId, const std::string& unitName,
                      Ts&&... args)
    {
        UnitId unitId{ UnitBaseType::Sink, 0, unitName };
        this->AddUnit(previousUnitId,
                      SharedPtr<T>(unitId, std::forward(args...)));
        return unitId;
    }


    virtual void Train(std::size_t epochs);

    virtual void Predict();

private:
    //! Helper function to create execution order of the execution graph
    //! \param unitId : Id of the unit to start searching
    //! \param depth : depth of the current
    void m_createExecutionOrder(UnitId unitId, std::size_t depth);
    std::vector<SharedPtr<ComputableUnit>> m_sourceUnitVector;
    std::vector<SharedPtr<ComputableUnit>> m_hiddenUnitVector;
    SharedPtr<SinkUnit> m_sinkUnit;
    //! CopyUnit mapped by unitId of each computableUnit
    std::unordered_map<UnitId, SharedPtr<CopyUnit>> m_copyUnitMap;
    //! Vector of grouped simultaneously executable units 
    std::vector<std::vector<UnitId>> m_executionOrder;
};
} // namespace CubbyDNN::Graph

#endif

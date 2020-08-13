// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_UNITMANAGER_DECL_HPP
#define TAKION_GRAPH_UNITMANAGER_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>
#include <Takion/FrontEnd/UnitMetaData.hpp>
#include <Takion/Computations/Optimizers/Optimizer.hpp>
#include <unordered_map>

namespace Takion::Graph
{
template <typename T>
class UnitManager
{
public:
    UnitManager(std::size_t batchSize)
        : m_batchSize(batchSize)
    {
    }

    virtual ~UnitManager() = default;

    UnitManager(const UnitManager<T>& unitManager) = delete;
    UnitManager(UnitManager<T>&& unitManager) noexcept;
    UnitManager<T>& operator=(const UnitManager<T>& unitManager) = delete;
    UnitManager<T>& operator=(UnitManager<T>&& unitManager) noexcept;

    void AppendUnit(UnitMetaData&& unitMetaData);

    Shape GetUnitOutputShape(const UnitId& unitId)
    {
        return m_unitMetaDataMap[unitId]->OutputShape();
    }

    void Compile(const std::string& optimizerName,
                 const Parameter& optimizerParameters);

    virtual void Forward(std::size_t cycle);

    virtual void Backward(std::size_t cycle);

    virtual void AsyncForward(std::size_t cycle);

    virtual void AsyncBackward(std::size_t cycle);

private:
    [[nodiscard]] bool m_isForwardCopyReady(const UnitId& subjectUnitId) const;
    [[nodiscard]] bool m_isBackwardCopyReady(const UnitId& subjectUnitId) const;
    //! Copies forward output of subject unit to forward inputs of destination units with direct connection
    void m_forwardCopy(const UnitId& subjectUnitId);
    //! Copies backward outputs of subject unit to backward inputs of destination units with direct connection
    void m_backwardCopy(const UnitId& subjectUnitId);
    //! Creates dependencies between output units and input units
    void m_connectUnits();

    [[nodiscard]] std::unique_ptr<Compute::Optimizer<T>> m_makeOptimizer(
        const std::string& optimizerName,
        const Parameter& parameters) const;

    std::unordered_map<UnitId, std::unique_ptr<UnitMetaData<T>>>
    m_unitMetaDataMap;
    std::unordered_map<UnitId, std::unique_ptr<ComputableUnit<T>>> m_unitMap;
    std::size_t m_batchSize;
};
} // namespace Takion::Graph

#endif

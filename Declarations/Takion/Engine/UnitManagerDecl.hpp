// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_UNITMANAGER_DECL_HPP
#define TAKION_GRAPH_UNITMANAGER_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>
#include <Takion/FrontEnd/UnitMetaData.hpp>
#include <Takion/Computations/Optimizers/Optimizer.hpp>
#include <Takion/Utils/Loaders/Loader.hpp>
#include <unordered_map>

namespace Takion::Engine
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

    FrontEnd::UnitMetaData<T>& GetUnitMetaData(const UnitId& unitId);

    void AppendUnit(FrontEnd::UnitMetaData<T>&& unitMetaData);

    void SetLoader(const UnitId& unitId,
                   std::unique_ptr<Util::Loader<T>> loader);

    Shape GetUnitOutputShape(const UnitId& unitId);

    void Compile(const std::string& optimizerName, const Parameter& parameter);

    virtual void Forward();

    virtual void Backward();

    virtual void AsyncForward(std::size_t cycle);

    virtual void AsyncBackward(std::size_t cycle);

    virtual void ResetState();

    virtual void ChangeBatchSize(std::size_t batchSize);

    [[nodiscard]] const Tensor<T>& GetOutput(UnitId unitId) const;

    std::unique_ptr<Graph::ComputableUnit<T>>& GetUnit(const UnitId& unitId);

private:
    [[nodiscard]] bool m_isForwardCopyReady(const UnitId& subjectUnitId) const;
    [[nodiscard]] bool m_isBackwardCopyReady(const UnitId& subjectUnitId) const;
    //! Copies forward output of subject unit to forward inputs of destination units with direct connection
    void m_forwardCopy(const UnitId& subjectUnitId);
    //! Copies backward outputs of subject unit to backward inputs of destination units with direct connection
    void m_backwardCopy(const UnitId& subjectUnitId);

    bool m_appendSource(const FrontEnd::UnitMetaData<T>& unitMetaData);
    bool m_appendHidden(const FrontEnd::UnitMetaData<T>& unitMetaData,
                        const std::string& optimizerName,
                        const Parameter& parameter);
    bool m_appendLoss(const FrontEnd::UnitMetaData<T>& unitMetaData);


    [[nodiscard]] std::unique_ptr<Compute::Optimizer<T>> m_makeOptimizer(
        const std::string& optimizerName,
        const Parameter& parameter) const;

    std::unordered_map<UnitId, FrontEnd::UnitMetaData<T>>
    m_unitMetaDataMap;
    std::unordered_map<UnitId, std::unique_ptr<Graph::ComputableUnit<T>>>
    m_unitMap;
    std::unordered_map<UnitId, std::unique_ptr<Util::Loader<T>>> m_loaderMap;
    std::size_t m_batchSize;
};
} // namespace Takion::Graph

#endif

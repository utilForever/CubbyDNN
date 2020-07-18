// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_UNITMANAGER_HPP
#define CUBBYDNN_UNITMANAGER_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <unordered_map>
#include <cubbydnn/Units/UnitMetadata.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/ActivationUnits/ActivationUnit.hpp>
#include <cubbydnn/Computations/LossFunctions/LossFunctions.hpp>


namespace CubbyDNN::Graph
{
class UnitManager
{
public:
    UnitManager() = default;
    virtual ~UnitManager() = default;

    UnitManager(const UnitManager& unitManager) = delete;
    UnitManager(UnitManager&& unitManager) noexcept;
    UnitManager& operator=(const UnitManager& unitManager) = delete;
    UnitManager& operator=(UnitManager&& unitManager) noexcept;

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
    bool m_isForwardCopyReady(const UnitId& subjectUnitId);
    void m_isBackwardCopyReady();
    //! Copies forward output of subject unit to forward inputs of destination units with direct connection
    void m_forwardCopy(const UnitId& subjectUnitId);
    //! Copies backward outputs of subject unit to backward inputs of destination units with direct connection
    void m_backwardCopy(const UnitId& subjectUnitId);
    //! Creates dependencies between output units and input units
    void m_connectUnits();

    [[nodiscard]] std::unique_ptr<Compute::Optimizer> m_makeOptimizer(
        const std::string& optimizerName,
        const Parameter& parameters) const;

    std::unordered_map<UnitId, std::unique_ptr<UnitMetaData>>
    m_unitMetaDataMap;
    std::unordered_map<UnitId, std::unique_ptr<ComputableUnit>> m_unitMap;
};
} // namespace CubbyDNN::Graph

#endif

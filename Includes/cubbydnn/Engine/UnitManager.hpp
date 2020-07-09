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

    Shape GetUnitOutputShape(std::size_t id)
    {
        return m_unitMetaDataMap[id]->OutputShape();
    }

    void Compile(std::string optimizerName,
                 const ParameterPack& optimizerParameters);

    virtual void Forward(std::size_t cycle);

    virtual void Backward(std::size_t cycle);

    virtual void AsyncForward(std::size_t cycle);

    virtual void AsyncBackward(std::size_t cycle);

private:
    //! Copies forward output of subject unit to forward inputs of destination units with direct connection
    void m_forwardCopy(std::size_t subjectUnitKey);
    //! Copies backward outputs of subject unit to backward inputs of destination units with direct connection
    void m_backwardCopy(std::size_t subjectUnitKey);

    void m_connectUnits();

    [[nodiscard]] std::unique_ptr<Compute::Optimizer> m_makeOptimizer(
        const std::string& optimizerName,
        const ParameterPack& parameters) const;

    std::unordered_map<std::size_t, std::unique_ptr<UnitMetaData>>
    m_unitMetaDataMap;
    std::unordered_map<std::size_t, std::unique_ptr<ComputableUnit>> m_unitMap;
};
} // namespace CubbyDNN::Graph

#endif

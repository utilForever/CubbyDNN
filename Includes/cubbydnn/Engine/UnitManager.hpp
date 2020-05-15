// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_UNITMANAGER_HPP
#define CUBBYDNN_UNITMANAGER_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <unordered_map>
#include <cubbydnn/Units/UnitMetadata.hpp>

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

    template <typename... Ts>
    void AppendUnit(const UnitMetaData& unitMetaData, Ts ... type);

    void Initialize();

    virtual void Forward(std::size_t cycle);

    virtual void Backward(std::size_t cycle);

    virtual void AsyncForward(std::size_t cycle);

    virtual void AsyncBackward(std::size_t cycle);

private:
    //! Copies output of source unit to destination units that has connection
    void m_forwardCopy(int sourceKey);

    void m_backwardCopy(int sourceKey);


    std::unordered_map<int, UnitMetaData> m_unitMetaDataMap;
    std::unordered_map<int, std::unique_ptr<ComputableUnit>> m_unitMap;
};
} // namespace CubbyDNN::Graph

#endif

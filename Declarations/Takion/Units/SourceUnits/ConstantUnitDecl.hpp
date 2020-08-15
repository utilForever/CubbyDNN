// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_CONSTANTUNIT_DECL_HPP
#define TAKION_GRAPH_CONSTANTUNIT_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>

namespace Takion::Graph
{
template <typename T>
class ConstantUnit : public ComputableUnit<T>
{
public:
    ConstantUnit(UnitId unitId, Tensor<T> tensor, std::size_t batchSize);
    ~ConstantUnit() override = default;

    ConstantUnit(const ConstantUnit<T>& constantUnit) = delete;
    ConstantUnit(ConstantUnit<T>&& constantUnit) noexcept;

    ConstantUnit& operator=(const ConstantUnit<T>& constantUnit) = delete;
    ConstantUnit& operator=(ConstantUnit<T>&& constantUnit) noexcept = default;

    static ConstantUnit<T> CreateUnit(
        const FrontEnd::UnitMetaData<T>& unitMetaData);

    void Forward() override
    {
    }

    void AsyncForward(std::promise<bool> promise) override
    {
        promise.set_value(true);
    }

    void Backward() override
    {
    }

    void AsyncBackward(std::promise<bool> promise) override
    {
        promise.set_value(true);
    }

};
}

#endif

// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_PLACEHOLDER_DECL_HPP
#define TAKION_GRAPH_PLACEHOLDER_DECL_HPP

#include <Takion/Units/ComputableUnit.hpp>
#include <Takion/Computations/Initializers/InitializerType.hpp>
#include <Takion/FrontEnd/UnitMetaData.hpp>
#include <functional>

namespace Takion::Graph
{
template <typename T>
class PlaceHolder : public ComputableUnit<T>
{
public:

    using ComputableUnit<T>::ForwardOutput;
    PlaceHolder(const UnitId& unitId, Tensor<T> forwardOutput,
                std::function<std::vector<T>()> loader, std::size_t batchSize);

    ~PlaceHolder() = default;

    static PlaceHolder<T> CreateUnit(
        const FrontEnd::UnitMetaData<T>& unitMetaData,
        std::function<std::vector<T>()>
        loader);


    PlaceHolder(const PlaceHolder& placeHolder) = delete;
    PlaceHolder(PlaceHolder&& placeHolder) noexcept = default;
    PlaceHolder& operator=(const PlaceHolder& placeHolder) = delete;
    PlaceHolder& operator=(PlaceHolder&& placeHolder) noexcept = default;

    void Forward() override;

    void AsyncForward(std::promise<bool> promise) override;

    void Backward() override;

    void AsyncBackward(std::promise<bool> promise) override;


private:
    std::function<std::vector<T>()> m_loader;
};
}
#endif

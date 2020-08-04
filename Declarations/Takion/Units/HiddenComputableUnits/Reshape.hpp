// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_GRAPH_RESHAPE_HPP
#define TAKION_GRAPH_RESHAPE_HPP
#include <Takion/Units/ComputableUnit.hpp>

namespace Takion::Graph
{
template <typename T>
class ReshapeUnit : public ComputableUnit<T>
{
    ReshapeUnit(UnitId unitId, NumberSystem numberSystem, Tensor forwardInput,
                Tensor backwardInput, Tensor forwardOutput,
                Tensor backwardOutput, Shape newShape)
        : ComputableUnit(std::move(unitId), numberSystem,
                         { std::move(forwardInput) },
                         { std::move(backwardInput) }, std::move(forwardOutput),
                         { std::move(backwardOutput) }
            )
    {
        if (forwardInput.TensorShape.Size() != newShape.Size())
            throw std::invalid_argument(
                "Size of new shape does not match original");
    }

    void Forward() override
    {
        Tensor<T>::ForwardTensorData(ComputableUnit<T>::ForwardInputVector.at(0),
                                  ComputableUnit<T>::ForwardOutput);
    }

    void Backward() override
    {
        Tensor<T>::ForwardTensorData(ComputableUnit<T>::BackwardOutputVector.at(0),
                                  ComputableUnit<T>::BackwardInputVector.at(0));
    }
};
}

#endif

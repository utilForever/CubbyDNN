// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef CUBBYDNN_RESHAPE_HPP
#define CUBBYDNN_RESHAPE_HPP
#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN::Graph
{
class ReshapeUnit : public ComputableUnit
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
        Tensor::CopyTensor(ForwardInputVector.at(0),
                           ForwardOutput);
    }

    void Backward() override
    {
        Tensor::CopyTensor(BackwardOutputVector.at(0),
                           BackwardInputVector.at(0));
    }
};
}

#endif

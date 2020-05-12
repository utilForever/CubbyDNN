// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.


#ifndef CUBBYDNN_RESHAPE_HPP
#define CUBBYDNN_RESHAPE_HPP
#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN::Graph
{
class Reshape : public ComputableUnit
{
    Reshape(UnitId unitId, Shape inputShape, Shape outputShape,
            NumberSystem numericType,
            std::size_t padSize = 0)
        : ComputableUnit(unitId, { std::move(inputShape) },
                         std::move(outputShape), numericType)
    {
    }

    void Forward() override
    {
        Tensor::CopyTensor(m_inputForwardTensorVector.at(0),
                           m_outputForwardTensor);
    }

    void Backward() override
    {
        Tensor::CopyTensor(m_inputBackwardTensor,
                           m_outputBackwardTensorVector.at(0));
    }
};
}

#endif

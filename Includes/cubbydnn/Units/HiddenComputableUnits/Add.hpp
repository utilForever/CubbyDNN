// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_ADD_HPP
#define CUBBYDNN_ADD_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>


namespace CubbyDNN::Graph
{
class AddUnit
{
public:
    AddUnit(UnitId unitId, NumberSystem numberSystem,
            std::vector<Tensor> forwardInputVector,
            std::vector<Tensor> backwardInputVector, Tensor forwardOutput,
            Tensor backwardOutput);
};
}


#endif

// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_BASEUNITS_HPP
#define CUBBYDNN_BASEUNITS_HPP

#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
class GenerateRandom
{
 public:
    explicit GenerateRandom(TensorInfo tensorInfo);

 private:
    TensorInfo m_tensorInfo;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_BASEUNITS_HPP

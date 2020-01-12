//
// Created by jwkim98 on 7/8/19.
//

#include <cubbydnn/Units/BaseUnits.hpp>

namespace CubbyDNN
{
GenerateRandom::GenerateRandom(TensorInfo tensorInfo)
    : m_tensorInfo(std::move(tensorInfo))
{
}
}  // namespace CubbyDNN
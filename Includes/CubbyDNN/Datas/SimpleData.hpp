#ifndef CUBBYDNN_SIMPLE_DATA_HPP
#define CUBBYDNN_SIMPLE_DATA_HPP

#include <CubbyDNN/Core/Memory.hpp>

namespace CubbyDNN
{
template <class InputT, class OutputT>
struct SimpleData
{
    using InputType = InputT;
    using OutputType = OutputT;

    InputType Data;
    OutputType Target;
};

struct SimpleBatch
{
    Core::Memory<float> Data;
    Core::Memory<float> Target;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_SIMPLE_DATA_HPP

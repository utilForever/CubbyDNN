#ifndef CUBBYDNN_SIMPLE_DATA_HPP
#define CUBBYDNN_SIMPLE_DATA_HPP

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
}  // namespace CubbyDNN

#endif  // CUBBYDNN_SIMPLE_DATA_HPP

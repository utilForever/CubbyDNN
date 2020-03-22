#ifndef CUBBYDNN_TRANSFORMS_HPP
#define CUBBYDNN_TRANSFORMS_HPP

namespace CubbyDNN::Transforms
{
template <class InputT, class OutputT>
class Transform
{
 public:
    using InputType = InputT;
    using OutputType = OutputT;

    virtual ~Transform() = default;

    virtual OutputType operator()(const InputType& input) = 0;
};
}  // namespace CubbyDNN::Transforms

#endif  // CUBBYDNN_TRANFORMS_HPP

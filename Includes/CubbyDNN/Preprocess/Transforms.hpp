#ifndef CUBBYDNN_TRANSFORMS_HPP
#define CUBBYDNN_TRANSFORMS_HPP

#include <CubbyDNN/Datas/Tensor.hpp>

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

class Normalize : public Transform<FloatTensor, FloatTensor>
{
public:
    Normalize(FloatTensor mean, FloatTensor var);

    OutputType operator()(const InputType& input) override;

private:
    FloatTensor mean_, var_;
};
}  // namespace CubbyDNN::Transforms

#endif  // CUBBYDNN_TRANFORMS_HPP

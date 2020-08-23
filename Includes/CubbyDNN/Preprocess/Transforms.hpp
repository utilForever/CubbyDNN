#ifndef CUBBYDNN_TRANSFORMS_HPP
#define CUBBYDNN_TRANSFORMS_HPP

#include <CubbyDNN/Core/Memory.hpp>
#include <CubbyDNN/Core/Shape.hpp>

#include <vector>

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

class Normalize : public Transform<Core::Memory<float>, Core::Memory<float>>
{
 public:
    using InputType = Core::Memory<float>;
    using OutputType = Core::Memory<float>;

    Normalize(std::vector<float> mean, std::vector<float> std,
              Core::Shape shape);
       

    OutputType operator()(const InputType& input) override;

 private:
    std::vector<float> mean_, std_;
    Core::Shape shape_;
};
}  // namespace CubbyDNN::Transforms

#endif  // CUBBYDNN_TRANFORMS_HPP

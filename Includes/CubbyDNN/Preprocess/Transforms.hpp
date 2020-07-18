#ifndef CUBBYDNN_TRANSFORMS_HPP
#define CUBBYDNN_TRANSFORMS_HPP

#include <CubbyDNN/Datas/SimpleData.hpp>
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

template <class Target>
class Normalize : public Transform<SimpleData<FloatTensor, Target>,
                                   SimpleData<FloatTensor, Target>>
{
 public:
    using DType = SimpleData<FloatTensor, Target>;
    using typename Transform<DType, DType>::InputType;
    using typename Transform<DType, DType>::OutputType;

    Normalize(FloatTensor mean, FloatTensor std)
        : mean_(std::move(mean)), std_(std::move(std))
    {
        // Do nothing
    }

    OutputType operator()(const InputType& input) override
    {
        OutputType t(input);

        for (auto& data : t.Data)
        {
            data = (data - mean_[0]) / std_[0];
        }

        return t;
    }

 private:
    FloatTensor mean_, std_;
};
}  // namespace CubbyDNN::Transforms

#endif  // CUBBYDNN_TRANFORMS_HPP

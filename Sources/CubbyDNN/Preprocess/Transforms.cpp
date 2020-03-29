#include <CubbyDNN/Preprocess/Transforms.hpp>

namespace CubbyDNN::Transforms
{
Normalize::Normalize(FloatTensor mean, FloatTensor var)
    : mean_(std::move(mean)), var_(std::move(var))
{
    // Do nothing
}

Normalize::OutputType Normalize::operator()(const InputType& input)
{
    return input;
}
}

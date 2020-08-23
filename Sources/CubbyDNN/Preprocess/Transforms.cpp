#include <CubbyDNN/Preprocess/Transforms.hpp>

#include <stdexcept>

namespace CubbyDNN::Transforms
{
Normalize::Normalize(std::vector<float> mean, std::vector<float> std,
                     Core::Shape shape)
    : mean_(std::move(mean)), std_(std::move(std)), shape_(std::move(shape))
{
    if (shape_.Rank() != 3)
        throw std::invalid_argument("Shape's rank must be 3");
}

Normalize::OutputType Normalize::operator()(const InputType& input)
{
    OutputType t(input);

    const std::size_t numOfChannels = shape_[2];
    const std::size_t sizeOfPlane = shape_[0] * shape_[1];

    for (std::size_t c = 0; c < numOfChannels; ++c)
    {
        for (std::size_t i = 0; i < sizeOfPlane; ++i)
        {
            auto& data = t.GetSpan()[i + c * sizeOfPlane];

            data = (data - mean_[c]) / std_[c];
        }
    }

    return t;
}
}

#ifndef CUBBYDNN_XAVIER_HPP
#define CUBBYDNN_XAVIER_HPP

#include <CubbyDNN/Initializer/Initializer.hpp>

#include <random>

namespace CubbyDNN::Initializer
{
class Xavier : public Initializer
{
 public:
    Xavier(std::mt19937_64::result_type seed, std::size_t fanIn,
           std::size_t fanOut);

    void operator()(Core::Span<float> span) override;

 private:
    std::mt19937_64 m_engine;
    std::normal_distribution<float> m_dist;
};
}  // namespace CubbyDNN::Initializer

#endif
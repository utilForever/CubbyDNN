#ifndef CUBBYDNN_CONSTANT_HPP
#define CUBBYDNN_CONSTANT_HPP

#include <CubbyDNN/Initializer/Initializer.hpp>

namespace CubbyDNN::Initializer
{
class Constant : public Initializer
{
 public:
    Constant(float constant = 0.0f);

    void operator()(Core::Span<float> span) override;

 private:
    float m_constant = 0.0f;
};
}  // namespace CubbyDNN::Initializer

#endif
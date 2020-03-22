#ifndef CUBBYDNN_DATASET_HPP
#define CUBBYDNN_DATASET_HPP

#include <tuple>

namespace CubbyDNN
{
template <class OutputT>
class Dataset
{
 public:
    using OutputType = OutputT;

    virtual ~Dataset() = default;

    virtual OutputType Get(std::size_t index) const = 0;
    virtual std::size_t Size() const = 0;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_DATASET_HPP

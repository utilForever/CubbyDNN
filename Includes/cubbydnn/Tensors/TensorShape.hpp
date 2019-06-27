// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_SHAPE_HPP
#define CUBBYDNN_TENSOR_SHAPE_HPP

#include <vector>

namespace CubbyDNN
{
//!
//! \brief TensorShape class.
//!
//! This class contains information about the shape of tensor.
//!
class TensorShape
{
 public:
    /// Constructs Empty TensorShape
    TensorShape() = default;
    /// Constructs TensorShape with given parameters
    TensorShape(long rows, long columns, long depth);

    bool operator==(const TensorShape& shape) const;
    bool operator!=(const TensorShape& shape) const;

    std::size_t Size() const noexcept;
    bool IsEmpty() const noexcept;

    /**
     * Remove row
     * @return
     */

    //TODO : remove row col depth
    //TODO : add Rank
    long Row() const;
    long Col() const;
    long Depth() const;

 private:

    std::vector<long> m_dimension;
    std::size_t m_totalSize = 0;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_SHAPE_HPP
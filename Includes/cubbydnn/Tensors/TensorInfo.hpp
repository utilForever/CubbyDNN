// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_SHAPE_HPP
#define CUBBYDNN_TENSOR_SHAPE_HPP

#include <functional>
#include <vector>

namespace CubbyDNN
{
enum class NumberSystem
{
    Float16,
    Float32,
    Double,
    Int8,
    Int16,
    Int32,
    Int64
};

//!
//! \brief TensorShape class.
//!
//! This class contains information about the shape of tensor.
//!

class TensorInfo
{
 public:
    /// Constructs TensorShape with given parameters
    TensorInfo(std::vector<size_t> shape, size_t unitByteSize,
                   NumberSystem numberSystem, bool isMutable = true);

    bool operator==(const TensorInfo& shape) const;
    bool operator!=(const TensorInfo& shape) const;

    /**
     * Size (Number of elements) of The TensorData
     * @return
     */
    size_t Size() const noexcept;
    size_t ByteSize() const noexcept;
    bool IsEmpty() const noexcept;
    const std::vector<size_t>& GetShape() const noexcept;

 private:
    std::function<size_t(const std::vector<size_t>)> m_getTotalByteSize;
    const std::vector<size_t> m_shape;
    const size_t m_totalSize;
    const size_t m_unitByteSize;
    const NumberSystem m_numberSystem;
    const bool m_isMutable;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_SHAPE_HPP
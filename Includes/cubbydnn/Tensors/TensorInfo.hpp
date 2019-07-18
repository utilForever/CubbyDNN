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

    /**
     * Brings back total byte size of the tensor
     * @return : total byte size of the tensor
     */
    size_t ByteSize() const noexcept;

    /**
     * Brings back whether tensor is empty or not
     * @return : true if empty otherwise false
     */
    bool IsEmpty() const noexcept;

    /**
     * Brings back number system of this tensor
     * @return
     */
    NumberSystem GetNumberSystem() const noexcept
    {
        return m_numberSystem;
    }

    /**
     * Brings back shape of this tensor
     * @return : shape of this tensor in vector
     */
    const std::vector<size_t>& GetShape() const noexcept;

 private:
    std::function<size_t(const std::vector<size_t>&)> m_getTotalByteSize;
    std::vector<size_t> m_shape;
    size_t m_totalSize;
    size_t m_unitByteSize;
    NumberSystem m_numberSystem;
    bool m_isMutable;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_SHAPE_HPP
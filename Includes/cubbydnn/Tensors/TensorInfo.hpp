// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_SHAPE_HPP
#define CUBBYDNN_TENSOR_SHAPE_HPP

#include <functional>
#include <map>
#include <vector>

namespace CubbyDNN
{
enum class NumberSystem
{
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Posit
};

//!
//! \brief TensorShape class.
//!
//! This class contains information about the shape of tensor.
//!

class TensorInfo
{
 public:
    static std::map<NumberSystem, size_t> UnitByteSizeMap;
    /// Constructs TensorShape with given parameters
    TensorInfo(std::vector<size_t> shape,
               NumberSystem numberSystem = NumberSystem::Float32,
               bool isMutable = true);

    bool operator==(const TensorInfo& shape) const;
    bool operator!=(const TensorInfo& shape) const;

    //!
    //! Size (Number of elements) of The TensorData
    //! @return
    [[nodiscard]] size_t Size() const noexcept;

    /**
     * Brings back total byte size of the tensor
     * @return : total byte size of the tensor
     */
    [[nodiscard]] size_t ByteSize() const noexcept;

    /**
     * Brings back whether tensor is empty or not
     * @return : true if empty otherwise false
     */
    [[nodiscard]] bool IsEmpty() const noexcept;

    /**
     * Brings back number system of this tensor
     * @return
     */
    [[nodiscard]] NumberSystem GetNumberSystem() const noexcept
    {
        return m_numberSystem;
    }

    [[nodiscard]] size_t GetNumberSystemByteSize() const noexcept{
        return m_unitByteSize;
    }

    /**
     * Brings back shape of this tensor
     * @return : shape of this tensor in vector
     */
    [[nodiscard]] const std::vector<size_t>& GetShape() const noexcept;

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
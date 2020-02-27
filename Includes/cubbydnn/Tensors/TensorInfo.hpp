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
};

enum class ShapeAlignment
{
    // Batch, Channel, height, Width
    NCHW,
    // Batch, Height Channel, Width
    NHCW,
    // Batch, Row, Column
    NRC,
    // Batch, Column, Row
    NCR,
    // Batch, Row
    NR,
    // Batch, Column
    NC,
    // Not specified
    None,
};

struct Shape
{
    size_t Batch = 0;
    size_t Channel = 0;
    size_t Row = 0;
    size_t Col = 0;


    friend bool operator==(const Shape& lhs, const Shape& rhs)
    {
        return lhs.Batch == rhs.Batch
               && lhs.Row == rhs.Row
               && lhs.Channel == rhs.Channel
               && lhs.Col == rhs.Col;
    }

    friend bool operator!=(const Shape& lhs, const Shape& rhs)
    {
        return !(lhs == rhs);
    }

    size_t GetTotalSize() const
    {
        return Batch * Row * Channel * Col;
    }
};

//! \brief TensorShape class.
//! This class contains information about the shape of tensor.
class TensorInfo
{
public:
    static std::map<NumberSystem, size_t> UnitByteSizeMap;
    /// Constructs TensorShape with given parameters
    TensorInfo(const Shape& shape, 
               NumberSystem numberSystem = NumberSystem::Float32);

    bool operator==(const TensorInfo& tensorInfo) const;
    bool operator!=(const TensorInfo& tensorInfo) const;

    //! GetSize (Number of elements) of The TensorData
    //! \return : Total element size of the tensor
    [[nodiscard]] size_t GetSize() const noexcept;

    //! Gets total byte size of the tensor
    //! \return : total byte size of the tensor
    [[nodiscard]] size_t GetByteSize() const noexcept;

    //! Gets number system of this tensor
    //! \return : NumberSystem of this tensor
    [[nodiscard]] NumberSystem GetNumberSystem() const noexcept
    {
        return m_numberSystem;
    }

    //! Gets shape of this tensor
    //! \return : shape of this tensor in vector
    [[nodiscard]] const Shape& GetShape() const noexcept;


private:
    Shape m_shape;
    size_t m_unitByteSize;
    NumberSystem m_numberSystem;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_SHAPE_HPP

// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_HPP
#define CUBBYDNN_TENSOR_DATA_HPP

#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cubbydnn/Computations/Device.hpp>

#include <cubbydnn/Utils/SharedPtr.hpp>

namespace CubbyDNN
{
//! TensorData class contains data vector for processing
//! with attributes describing it
class Tensor
{
public:
    Tensor() = default;
    Tensor(Shape shape, Compute::Device device,
           NumberSystem numberSystem = NumberSystem::Float);

    ~Tensor();

    Tensor(const Tensor& tensor);
    Tensor(Tensor&& tensor) noexcept;
    /// move assignment operator
    Tensor& operator=(const Tensor& tensor);
    Tensor& operator=(Tensor&& tensor) noexcept;

    //! If both tensors are on same device, data is moved rather than copied
    static void ForwardTensor(const Tensor& source, Tensor& destination);

    static void MoveTensor(const Tensor& source, Tensor& destination);

    static void CopyTensor(const Tensor& source, Tensor& destination);

    template <typename T>
    T& At(std::vector<std::size_t> index)
    {
        if (index.size() != TensorShape.Dim())
            throw std::invalid_argument(
                "Index must have same dimension with tensor shape");
        const int columnIdx = static_cast<int>(TensorShape.Dim() - 1);
        int shapeIdx = columnIdx;
        int idx = columnIdx;
        std::size_t multiplier = 1;
        std::size_t offset = 0;
        for (; shapeIdx >= 0 && idx != static_cast<int>(index.size());
               --shapeIdx, --idx)
        {
            offset += multiplier * index.at(idx);
            if (idx == columnIdx && Device.PadSize() > 0)
                multiplier = numPaddedColumn;
            else
                multiplier *= TensorShape.At(idx);
        }

        return *(static_cast<T*>(DataPtr) + offset);
    }

    std::size_t GetColumnElementSize() const
    {
        return numPaddedColumn;
    }

    /// Data vector which possesses actual data
    void* DataPtr;
    /// Shape of this tensorData
    Shape TensorShape;
    NumberSystem NumericType = NumberSystem::Float;
    Compute::Device Device;
    std::atomic<std::size_t> ForwardState = 0;
    std::atomic<std::size_t> BackwardState = 0;

private:
    std::size_t numPaddedColumn = 0;

    [[nodiscard]] std::size_t getDataSize()
    {
        std::size_t size = 1;
        for(std::size_t i = 0; i < TensorShape.Dim() - 1; ++i)
        {
            size *= i;
        }
        return size * m_getPaddedColumnSize();
    }

    std::size_t m_getPaddedColumnSize() const
    {
        if (Device.PadSize() == 0)
            return TensorShape.NumCols();

        std::size_t padUnitSize;
        if (NumericType == NumberSystem::Float)
            padUnitSize = Device.PadSize() / sizeof(float);
        else
            padUnitSize = Device.PadSize() / sizeof(int);

        std::size_t i = 0;
        while (padUnitSize * i < TensorShape.NumCols())
            ++i;

        return padUnitSize * i;
    }
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP

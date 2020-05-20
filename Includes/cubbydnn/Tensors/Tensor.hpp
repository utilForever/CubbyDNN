// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_DATA_HPP
#define CUBBYDNN_TENSOR_DATA_HPP

#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cubbydnn/Computations/Device.hpp>

namespace CubbyDNN
{
//! TensorData class contains data vector for processing
//! with attributes describing it
class Tensor
{
public:
    Tensor(Shape shape, Compute::Device device,
           NumberSystem numberSystem = NumberSystem::Float);

    ~Tensor();

    Tensor(const Tensor& tensor);
    Tensor(Tensor&& tensor) noexcept;
    /// move assignment operator
    Tensor& operator=(const Tensor& tensor) = delete;
    Tensor& operator=(Tensor&& tensor) noexcept;

    static void CopyTensor(const Tensor& source, Tensor& destination);

    template <typename T>
    T& At(std::vector<std::size_t> index)
    {
        std::size_t shapeIdx = 0;
        std::size_t idx = 0;
        std::size_t multiplier = 1;
        std::size_t offset = 0;
        for (; shapeIdx != TensorShape.Dim() && idx != index.size();
               ++shapeIdx, ++idx)
        {
            offset += multiplier * index.at(idx);
            if (idx == 0 && Device.PadSize() > 0)
                multiplier = numPaddedColumn;
            else
                multiplier *= TensorShape.At(idx);
        }

        return *(static_cast<T*>(DataPtr) + offset);
    }

    std::size_t GetPaddedNumCols() const
    {
        return numPaddedColumn;
    }

    /// Data vector which possesses actual data
    void* DataPtr = nullptr;
    /// Shape of this tensorData
    Shape TensorShape;
    NumberSystem NumericType = NumberSystem::Float;
    Compute::Device Device;
    std::atomic<std::size_t> ForwardStateNum = 0;
    std::atomic<std::size_t> BackwardStateNum = 0;

private:
    std::size_t numPaddedColumn = 0;

    template <typename T>
    std::size_t m_getPaddedColumnSize() const
    {
        if (Device.PadSize() == 0)
            return TensorShape.NumCols();

        const auto padUnitSize = Device.PadSize() / sizeof(T);

        std::size_t i = 0;
        while (padUnitSize * i < TensorShape.NumCols())
            ++i;

        return padUnitSize * i;
    }
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP

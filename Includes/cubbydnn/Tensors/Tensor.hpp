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

    Tensor(Shape shape, Compute::Device device, std::vector<float> data);
    Tensor(Shape shape, Compute::Device device, std::vector<int> data);

    ~Tensor();

    Tensor(const Tensor& tensor);
    Tensor(Tensor&& tensor) noexcept;
    /// move assignment operator
    Tensor& operator=(const Tensor& tensor);
    Tensor& operator=(Tensor&& tensor) noexcept;

    //! If both tensors are on same device, data is moved rather than copied
    static void ForwardTensorData(Tensor& source, Tensor& destination);

    static void MoveTensorData(Tensor& source, Tensor& destination);

    static void CopyTensorData(const Tensor& source, Tensor& destination);

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
                multiplier = m_paddedColumnSize;
            else
                multiplier *= TensorShape.At(idx);
        }
        T& val = *(static_cast<T*>(DataPtr) + offset);
        return val;
    }

    std::size_t GetColumnElementSize() const
    {
        return m_paddedColumnSize;
    }

    [[nodiscard]] std::size_t GetElementSize() const
    {
        std::size_t size = 1;
        for (std::size_t i = 0; i < TensorShape.Dim() - 1; ++i)
        {
            size *= TensorShape.At(i);
        }
        return size * m_getPaddedColumnSize();
    }

    [[nodiscard]] std::size_t GetDataByteSize() const
    {
        if (NumericType == NumberSystem::Float)
            return GetElementSize() * sizeof(float);
        return GetElementSize() * sizeof(int);
    }

    /// Data vector which possesses actual data
    void* DataPtr = nullptr;
    /// Shape of this tensorData
    Shape TensorShape;
    Compute::Device Device;
    NumberSystem NumericType;

    std::atomic<std::size_t> State = 0;

private:
    std::size_t m_paddedColumnSize = 0;
    std::atomic<bool> m_hasOwnership = false;


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

    void m_freeData()
    {
        if (m_hasOwnership)
        {
            m_hasOwnership.exchange(false, std::memory_order_acquire);
            if (NumericType == NumberSystem::Float)
                delete[] static_cast<float*>(DataPtr);
            else
                delete[] static_cast<int*>(DataPtr);
        }
    }
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_DATA_HPP

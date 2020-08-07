// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_TENSOR_DECL_HPP
#define TAKION_TENSOR_DECL_HPP

#include <Takion/Computations/Device.hpp>
#include <Takion/Utils/Shape.hpp>
#include <atomic>
#include <Takion/Utils/Span.hpp>

namespace Takion
{
//! TensorData class contains data vector for processing
//! with attributes describing it
template <typename T>
class Tensor
{
public:
    Tensor(Shape shape, Compute::Device device);

    Tensor(Shape shape, std::size_t batchSize, Compute::Device device);

    Tensor(Shape shape, std::size_t batchSize, Compute::Device device,
           std::vector<T> data);

    ~Tensor();

    Tensor(const Tensor<T>& tensor);
    Tensor(Tensor<T>&& tensor) noexcept;
    /// move assignment operator
    Tensor<T>& operator=(const Tensor<T>& tensor);
    Tensor<T>& operator=(Tensor<T>&& tensor) noexcept;

    [[nodiscard]] Tensor<T> SubTensor(std::initializer_list<int> index);

    //! If both tensors are on same device, data is moved rather than copied
    static void ForwardTensorData(Tensor<T>& source, Tensor<T>& destination);

    static void MoveTensorData(Tensor<T>& source, Tensor<T>& destination);

    static void CopyTensorData(const Tensor<T>& source, Tensor<T>& destination);

    T& At(std::size_t batchIdx, std::vector<std::size_t> index);

    std::size_t ColumnElementSize() const
    {
        return m_getPaddedColumnSize();
    }

    [[nodiscard]] std::size_t ElementSize() const
    {
        std::size_t size = 1;
        for (std::size_t i = 0; i < TensorShape.Dim() - 1; ++i)
        {
            size *= TensorShape.At(i);
        }
        return size * m_getPaddedColumnSize();
    }

    [[nodiscard]] std::size_t BatchElementSize() const
    {
        std::size_t size = 1;
        for (std::size_t i = 0; i < TensorShape.Dim() - 1; ++i)
        {
            size *= TensorShape.At(i);
        }
        return size * m_getPaddedColumnSize() * BatchSize;
    }

    [[nodiscard]] std::size_t GetDataByteSize() const
    {
        return BatchElementSize() * sizeof(T);
    }

    /// Data vector which possesses actual data
    Utils::Span<T> Data;
    /// Shape of this tensorData
    Shape TensorShape;
    Compute::Device Device;
    std::size_t BatchSize = 0;
    std::atomic<std::size_t> State = 0;

private:
    std::size_t m_paddedColumnSize = 0;
    std::atomic<bool> m_hasOwnership = false;


    std::size_t m_getPaddedColumnSize() const
    {
        if (Device.PadSize() == 0)
            return TensorShape.NumCol();

        const std::size_t padUnitSize = Device.PadSize() / sizeof(T);

        std::size_t i = 0;
        while (padUnitSize * i < TensorShape.NumCol())
            ++i;

        return padUnitSize * i;
    }

    void m_freeData()
    {
        if (m_hasOwnership)
        {
            m_hasOwnership.exchange(false, std::memory_order_acquire);
            Data.Clear();
        }
    }
};
} // namespace Takion

#endif  // Takion_TENSOR_DATA_HPP

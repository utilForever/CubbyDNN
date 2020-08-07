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

    [[nodiscard]] std::size_t ColumnElementSize() const
    {
        return m_paddedColumnSize;
    }

    [[nodiscard]] std::size_t ElementSize() const
    {
        return m_elementSize;
    }

    [[nodiscard]] std::size_t BatchElementSize() const
    {
        return m_elementSize * BatchSize;
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
    std::size_t m_elementSize = 0;
    std::size_t m_paddedColumnSize = 0;
    std::atomic<bool> m_hasOwnership = false;

    std::size_t m_getElementSize() const;

    std::size_t m_getPaddedColumnSize() const;

    void m_freeData();
};
} // namespace Takion

#endif  // Takion_TENSOR_DATA_HPP

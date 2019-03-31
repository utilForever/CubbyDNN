// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_OBJECT_HPP
#define CUBBYDNN_TENSOR_OBJECT_HPP

#include <cubbydnn/Tensors/Decl/TensorData.hpp>
#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cubbydnn/Tensors/TensorShape.hpp>
#include <cubbydnn/Tensors/Decl/TensorSocket.hpp>

#include <memory>
#include <mutex>

namespace CubbyDNN
{
//!
//! \brief TensorObject class.
//!
//! This class will represent graph at runtime, and stores actual data used in
//! graph execution.
//!
template <typename T>
class TensorObject
{
 public:
    TensorObject<T>(std::size_t size, TensorShape shape, long from, long to);
    ~TensorObject<T>() = default;
    TensorObject<T>(const TensorObject& obj);
    TensorObject<T>(TensorObject&& obj) noexcept;
    TensorObject<T>& operator=(const TensorObject<T>& obj);
    TensorObject<T>& operator=(TensorObject<T>&& obj) noexcept;

    bool operator==(const TensorObject<T>& obj) const;

    const TensorInfo& Info() const;
    std::vector<T> Data() const;
    std::unique_ptr<TensorData<T>> DataPtr();

    TensorShape DataShape() const;

    void MakeImmutable() const;

 private:
    /// Includes information about this TensorObject
    TensorInfo m_info;
    /// ptr to Data this TensorObject holds
    std::unique_ptr<TensorData<T>> m_data;
    /// mtx for accessing the data
    std::mutex m_dataMtx;
    /// ptr to next operation
    std::unique_ptr<TensorSocket<T>> m_nextOperation;
};
}  // namespace CubbyDNN


#endif  // CUBBYDNN_TENSOR_OBJECT_HPP
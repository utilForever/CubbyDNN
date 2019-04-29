// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_OBJECT_HPP
#define CUBBYDNN_TENSOR_OBJECT_HPP

#include <cubbydnn/Tensors/Decl/TensorData.hpp>
#include <cubbydnn/Tensors/Decl/TensorSocket.hpp>
#include <cubbydnn/Tensors/TensorInfo.hpp>
#include <cubbydnn/Tensors/TensorShape.hpp>

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
class TensorPlug
{
 public:
    explicit TensorPlug<T>(const TensorShape& shape,
                           TensorSocketPtr<T> tensorSocketPtr);
    explicit TensorPlug<T>(const TensorInfo& tensorInfo,
                           TensorSocketPtr<T> tensorSocketPtr);

    /// Only move constructor is allowed
    TensorPlug<T>(TensorPlug&& obj) noexcept;

    /// Only move assign operator is allowed
    TensorPlug<T>& operator=(TensorPlug<T>&& obj) noexcept;

    /**
     * Gets information object that describes this TensorObject
     * @return : TensorInfo object describing this TensorObject
     */
    const TensorInfo& Info() const noexcept;

    bool SendData();

    TensorDataPtr<T> GetDataPtr() const noexcept;

    bool SetDataPtr(TensorDataPtr<T> tensorDataPtr);

    std::future<TensorData<T>> GetFuture();

 private:
    /// Includes information about this TensorObject
    TensorInfo m_info;
    /// ptr to Data this TensorObject holds
    TensorDataPtr<T> m_data = nullptr;
    /// TensorSocket that this TensorObject is connected
    std::unique_ptr<TensorSocket<T>> m_socket;

    std::promise<TensorDataPtr<T>> m_promise;
};

template <typename T>
using TensorPlugPtr = typename std::unique_ptr<TensorPlug<T>>;
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_OBJECT_HPP
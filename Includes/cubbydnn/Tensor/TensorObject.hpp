// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_OBJECT_HPP
#define CUBBYDNN_TENSOR_OBJECT_HPP

#include <cubbydnn/Tensor/TensorData.hpp>
#include <cubbydnn/Tensor/TensorInfo.hpp>
#include <cubbydnn/Tensor/TensorShape.hpp>

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
class TensorObject
{
 public:
    TensorObject(std::size_t size, TensorShape shape, long from, long to);
    ~TensorObject() = default;
    TensorObject(const TensorObject& obj);
    TensorObject(TensorObject&& obj) noexcept;
    TensorObject& operator=(const TensorObject& obj);
    TensorObject& operator=(TensorObject&& obj) noexcept;

    bool operator==(const TensorObject& obj) const;

    const TensorInfo& Info() const;
    std::vector<float> Data() const;
    std::unique_ptr<TensorData> DataPtr();
    TensorShape DataShape() const;

    void MakeImmutable() const;

 private:
    TensorInfo m_info;
    std::unique_ptr<TensorData> m_data;
    std::mutex m_dataMtx;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_OBJECT_HPP
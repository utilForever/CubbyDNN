// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_HPP
#define CUBBYDNN_TENSOR_HPP

#include <cubbydnn/Tensors/TensorShape.hpp>

namespace CubbyDNN
{
//!
//! \brief Tensor class.
//!
//! This graph contains information of graph being built methods of operation
//! class builds TensorData class based on information of this class.
//!
class Tensor
{
 public:
    Tensor(TensorShape shape, long prevOpID, bool isMutable = true) noexcept;

    const TensorShape& Shape() const noexcept;
    std::size_t DataSize() const noexcept;
    long PrevOpID() const noexcept;

    void AddOp(long nextOpID);

    bool IsValid() const noexcept;
    bool IsMutable() const noexcept;

    void MakeImmutable() noexcept;

 private:
    //! Shape of this tensor represents.
    TensorShape m_shape;

    //! Previous operation ID of operation that this tensor is generated.
    long m_prevOpID;

    //! Container for storing operations this tensor will head to.
    std::vector<long> m_nextOps;

    //! Determines whether data of this tensor can be modified.
    bool m_isMutable = true;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_HPP
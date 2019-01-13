// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_TENSOR_INFO_HPP
#define CUBBYDNN_TENSOR_INFO_HPP

namespace CubbyDNN
{
//!
//! \brief TensorInfo class.
//!
class TensorInfo
{
 public:
    TensorInfo() = default;
    TensorInfo(long from, long to, bool isMutable = true);

 private:
    long m_from;
    long m_to;

    bool m_busy = false;
    unsigned m_processCount = 0;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_TENSOR_INFO_HPP
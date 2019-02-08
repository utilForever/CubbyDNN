// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_OPERATION_HPP
#define CUBBYDNN_OPERATION_HPP

#include <cubbydnn/Operations/OpEnums.hpp>
#include <cubbydnn/Operations/OperationInfo.hpp>

#include <string>
#include <vector>

namespace CubbyDNN
{
//!
//! \brief Operation class.
//!
//! This class is contained in graph structure. It contains information about
//! which operation to execute, tensors that data comes from and tensor to
//! output processed data.
//!
class Operation
{
 public:
    const std::string& GetName();

    long GetID();

    const std::vector<long>& GetInputTensors() const;

    const std::vector<long>& GetOutputTensors() const;

    void AddInputTensor(long tensorID);

    void AddOutputTensor(long tensorID);

    std::size_t GetNumOfInputTensors();

    std::size_t GetNumOfOutputTensors();

    unsigned GetProcessCount();

    void IncrementProcessCount();

    OperationInfo GetInfo() const;

 protected:
    explicit Operation() = default;

    std::string m_name;
    OpType m_type = OpType::EMPTY;
    long m_id = 0;

    std::vector<long> m_vecInputTensorID;
    std::vector<long> m_vecOutputTensorID;

    unsigned m_processCount = 0;
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_OPERATION_HPP
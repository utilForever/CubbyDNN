// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Operations/Decl/Operation.hpp>

namespace CubbyDNN
{
const std::string& Operation::GetName() const noexcept
{
    return m_name;
}

long Operation::GetID() const noexcept
{
    return m_id;
}

const std::vector<long>& Operation::GetInputTensors() const noexcept
{
    return m_vecInputTensorID;
}

const std::vector<long>& Operation::GetOutputTensors() const noexcept
{
    return m_vecOutputTensorID;
}

void Operation::AddInputTensor(long tensorID)
{
    m_vecInputTensorID.emplace_back(tensorID);
}

void Operation::AddOutputTensor(long tensorID)
{
    m_vecOutputTensorID.emplace_back(tensorID);
}

std::size_t Operation::GetNumOfInputTensors() const noexcept
{
    return m_vecInputTensorID.size();
}

std::size_t Operation::GetNumOfOutputTensors() const noexcept
{
    return m_vecOutputTensorID.size();
}

unsigned Operation::GetProcessCount() const noexcept
{
    return m_processCount;
}

void Operation::IncrementProcessCount() noexcept
{
    m_processCount += 1;
}

OperationInfo Operation::GetInfo() const
{
    return OperationInfo(m_id, m_name, m_vecInputTensorID.size(),
                         m_vecOutputTensorID.size());
}
}  // namespace CubbyDNN
// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Operations/Operation.hpp>

namespace CubbyDNN
{
const std::string& Operation::GetName()
{
    return m_name;
}

long Operation::GetID()
{
    return m_id;
}

const std::vector<long>& Operation::GetInputTensors() const
{
    return m_vecInputTensorID;
}

const std::vector<long>& Operation::GetOutputTensors() const
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

std::size_t Operation::GetNumOfInputTensors()
{
    return m_vecInputTensorID.size();
}

std::size_t Operation::GetNumOfOutputTensors()
{
    return m_vecOutputTensorID.size();
}

unsigned Operation::GetProcessCount()
{
    return m_processCount;
}

void Operation::IncrementProcessCount()
{
    m_processCount += 1;
}

OperationInfo Operation::GetInfo() const
{
    return OperationInfo(m_id, m_name, m_vecInputTensorID.size(),
                         m_vecOutputTensorID.size());
}
}  // namespace CubbyDNN
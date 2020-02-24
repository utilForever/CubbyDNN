// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/SinkComputableUnits/SinkUnit.hpp>
#include <iostream>

namespace CubbyDNN
{
SinkUnit::SinkUnit(std::vector<TensorInfo> inputTensorInfoVector)
    : ComputableUnit(inputTensorInfoVector, {}, UnitType::Sink)
{
    m_inputPtrVector =
        std::vector<SharedPtr<ComputableUnit>>(m_inputTensorInfoVector.size());

    m_inputTensorVector.reserve(m_inputTensorInfoVector.size());
    for (const auto& tensorInfo : m_inputTensorInfoVector)
    {
        m_inputTensorVector.emplace_back(AllocateTensor(tensorInfo));
    }
}

bool SinkUnit::IsReady()
{
    if (m_unitState.IsBusy)
        return false;

    for (const auto& previousPtr : m_inputPtrVector)
    {
        if (previousPtr->GetStateNum() != GetStateNum() + 1)
            return false;
    }
    return true;
}

void SinkUnit::Compute()
{
    std::cout << "SinkUnit at epoch : " << m_unitState.StateNum << std::endl;
}

SinkTestUnit::SinkTestUnit(
    std::vector<TensorInfo> inputTensorInfoVector,
    std::function<void(const Tensor&)> testFunction)
    : SinkUnit(std::move(inputTensorInfoVector)),
      m_testFunction(std::move(testFunction))
{
}

void SinkTestUnit::Compute()
{
    std::cout << "SinkUnit at epoch : " << m_unitState.StateNum << std::endl;
    m_testFunction(m_inputTensorVector.at(0));
}
} // namespace CubbyDNN

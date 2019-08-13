//
// Created by jwkim98 on 8/13/19.
//

#include <cubbydnn/Units/CopyUnit.hpp>

namespace CubbyDNN{

    CopyUnit::CopyUnit() : ComputableUnit(1, 1)
    {
    }

    CopyUnit::CopyUnit(CopyUnit&& copyUnit) noexcept
            : ComputableUnit(std::move(copyUnit))
    {
    }

    void CopyUnit::Compute()
    {
        std::cout << "CopyUnit" << std::endl;
        std::cout << m_unitState.StateNum << std::endl;

        auto& inputTensor =
                m_inputPtrVector.at(0)->GetOutputTensor(m_inputTensorIndex);

        auto& outputTensor =
                m_outputPtrVector.at(0)->GetInputTensor(m_outputTensorIndex);
        CopyTensor(inputTensor, outputTensor);
    }

    bool CopyUnit::IsReady()
    {
        if (ComputableUnit::m_unitState.IsBusy)
            return false;

        auto& stateNum = GetStateNum();
        return (ComputableUnit::m_inputPtrVector.at(0)->GetStateNum() ==
                (stateNum + 1) &&
                ComputableUnit::m_outputPtrVector.at(0)->GetStateNum() == stateNum);
    }
}


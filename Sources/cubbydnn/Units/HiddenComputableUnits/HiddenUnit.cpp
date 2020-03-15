// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>

namespace CubbyDNN
{
HiddenUnit::HiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
                       TensorInfo outputTensorInfo, std::size_t numberOfOutputs)
    : ComputableUnit(std::move(inputTensorInfoVector),
                     std::move(outputTensorInfo),
                     UnitType::Hidden)
{
    m_outputPtrVector = std::vector<SharedPtr<ComputableUnit>>(numberOfOutputs);
    m_inputPtrVector =
        std::vector<SharedPtr<ComputableUnit>>(m_inputTensorInfoVector.size());

    m_inputTensorVector.reserve(m_inputTensorInfoVector.size());
    for (auto& inputTensorInfo : m_inputTensorInfoVector)
    {
        m_inputTensorVector.emplace_back(AllocateTensor(inputTensorInfo));
    }

    m_outputTensorVector.reserve(numberOfOutputs);
    for (std::size_t idx = 0; idx < numberOfOutputs; ++idx)
    {
        m_outputTensorVector.emplace_back(AllocateTensor(m_outputTensorInfo));
    }
}

HiddenUnit::HiddenUnit(HiddenUnit&& hiddenUnit) noexcept
    : ComputableUnit(std::move(hiddenUnit))
{
    m_tensorOperation = std::move(hiddenUnit.m_tensorOperation);
}

HiddenUnit& HiddenUnit::operator=(HiddenUnit&& hiddenUnit) noexcept
{
    m_tensorOperation = std::move(hiddenUnit.m_tensorOperation);
    ComputableUnit::operator=(std::move(hiddenUnit));
    return *this;
}

bool HiddenUnit::IsReady()
{
    for (auto& elem : m_inputPtrVector)
    {
        if (elem->GetStateNum() != this->GetStateNum() + 1)
            return false;
    }

    for (auto& elem : m_outputPtrVector)
    {
        if (elem->GetStateNum() != this->GetStateNum())
            return false;
    }

    return true;
}

MatMul::MatMul(const TensorInfo& inputA, const TensorInfo& inputB,
               const TensorInfo& output, std::size_t numberOfOutputs)
    : HiddenUnit({ inputA, inputB }, output, numberOfOutputs)
{
    auto& shapeA = inputA.GetShape();
    auto& shapeB = inputB.GetShape();

    if (shapeA.Dim() != shapeB.Dim())
        throw std::runtime_error("Multiply - dimension mismatch");

    if (shapeA.Dim() > 1)
    {
        if (shapeA.Col() != shapeB.Row())
            throw std::runtime_error(
                "Multiply- number of columns of A and number of rows of B must "
                "be "
                "identical");

        for (std::size_t i = 0; i < shapeA.Dim() - 2; ++i)
            if (shapeA.At(i) != shapeB.At(i))
                throw std::runtime_error("Multiply -shape mismatch");
    }
    else
        throw std::runtime_error(
            "Multiply - Dimension must be equal or greater than 2");
}

void MatMul::Compute()
{
    m_tensorOperation->Multiply(m_inputTensorVector.at(0),
                                m_inputTensorVector.at(1),
                                m_outputTensorVector.at(0));
}
} // namespace CubbyDNN

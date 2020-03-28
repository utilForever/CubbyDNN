// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cassert>
#include <cubbydnn/Units/SourceComputableUnits/SourceUnit.hpp>

namespace CubbyDNN
{
SourceUnit::SourceUnit(TensorInfo output)
    : ComputableUnit(UnitType::Source, {}, std::move(output))
{
    m_outputForwardTensor = AllocateTensor(m_outputTensorInfo);
}

SourceUnit::SourceUnit(SourceUnit&& sourceUnit) noexcept
    : ComputableUnit(std::move(sourceUnit))
{
}

SourceUnit& SourceUnit::operator=(SourceUnit&& sourceUnit) noexcept
{
    if (this == &sourceUnit)
        return *this;
    ComputableUnit::operator=(std::move(sourceUnit));
    return *this;
}

ConstantUnit::ConstantUnit(TensorInfo output, void* dataPtr)
    : SourceUnit(std::move(output)), m_dataPtr(dataPtr)
{
    const auto byteSize = output.GetByteSize();
    assert(dataPtr != nullptr);
    m_byteSize = byteSize;

    std::memcpy(m_outputForwardTensor.DataPtr, static_cast<void*>(m_dataPtr),
                m_byteSize);
}

ConstantUnit::~ConstantUnit()
{
    free(m_dataPtr);
}

ConstantUnit::ConstantUnit(ConstantUnit&& constantUnit) noexcept
    : SourceUnit(std::move(constantUnit)),
      m_dataPtr(constantUnit.m_dataPtr),
      m_byteSize(constantUnit.m_byteSize)
{
}

ConstantUnit& ConstantUnit::operator=(ConstantUnit&& constantUnit) noexcept
{
    if (this != &constantUnit)
    {
        m_dataPtr = constantUnit.m_dataPtr;
        m_byteSize = constantUnit.m_byteSize;
        SourceUnit::operator=(std::move(constantUnit));
    }
    return *this;
}

void ConstantUnit::Forward()
{
    m_outputForwardTensor = AllocateTensor(m_outputTensorInfo);
    m_outputForwardTensor.DataPtr = m_dataPtr;
}
}  // namespace CubbyDNN

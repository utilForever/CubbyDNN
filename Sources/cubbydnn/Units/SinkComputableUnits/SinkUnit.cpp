// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/SinkComputableUnits/SinkUnit.hpp>

namespace CubbyDNN
{
SinkUnit::SinkUnit(std::vector<TensorInfo> inputTensorInfoVector)
    : ComputableUnit(UnitType::Sink, std::move(inputTensorInfoVector),
                     TensorInfo())
{
    m_inputPtrVector =
        std::vector<SharedPtr<ComputableUnit>>(m_inputTensorInfoVector.size());

    m_inputForwardTensorVector.reserve(m_inputTensorInfoVector.size());
    for (const auto& tensorInfo : m_inputTensorInfoVector)
    {
        m_inputForwardTensorVector.emplace_back(AllocateTensor(tensorInfo));
    }
}

SinkUnit::SinkUnit(SinkUnit&& sinkUnit) noexcept
    : ComputableUnit(std::move(sinkUnit))
{
}

SinkUnit& SinkUnit::operator=(SinkUnit&& sinkUnit) noexcept
{
    if (this != &sinkUnit)
        ComputableUnit::operator=(std::move(sinkUnit));
    return *this;
}

} // namespace CubbyDNN

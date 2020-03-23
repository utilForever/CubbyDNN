// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SINKUNIT_HPP
#define CUBBYDNN_SINKUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Utils/SharedPtr.hpp>

#include <functional>

namespace CubbyDNN
{
//! Unit that has no output, but has inputs
//! This type of unit plays role as sink of the computable graph
class SinkUnit : public ComputableUnit
{
public:
    //! Constructor
    //! \param inputTensorInfoVector : vector of tensorInfo to accept
    explicit SinkUnit(std::vector<TensorInfo> inputTensorInfoVector);
    ~SinkUnit() = default;

    //! SinkUnit is not copy-assignable
    SinkUnit(const SinkUnit& sinkUnit) = delete;
    SinkUnit(SinkUnit&& sinkUnit) noexcept;

    //! SinkUnit is not copy-assignable
    SinkUnit& operator=(const SinkUnit& sinkUnit) = delete;
    SinkUnit& operator=(SinkUnit&& sinkUnit) noexcept;

    //! Brings back if executableUnit is ready to be executed
    //! \return : whether corresponding unit is ready to be executed
    bool IsReady() final;

    std::size_t AddInputPtr(const SharedPtr<ComputableUnit>& computableUnitPtr,
                            std::size_t index);

    virtual void Forward()
    {
    }

    virtual void Backward()
    {
    }

private:
    /// ptr to units to receive result from
    std::vector<SharedPtr<ComputableUnit>> m_inputPtrVector;
    //! vector of tensor information for input in forward propagation
    std::vector<TensorInfo> m_inputTensorInfoVector;
    //! vector of input tensors used to compute forward propagation
    std::vector<Tensor> m_inputForwardTensorVector;
    //! single output tensor of back propagation
    std::vector<Tensor> m_outputBackwardTensorVector;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_SINKUNIT_HPP

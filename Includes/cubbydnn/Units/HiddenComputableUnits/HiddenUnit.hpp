/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_HIDDENUNIT_HPP
#define CUBBYDNN_HIDDENUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Computations/TensorOperations/TensorOperations.hpp>
#include <cubbydnn/Computations/TensorOperations/NaiveOperations.hpp>
#include <cubbydnn/Utils/SharedPtr.hpp>
#include <iostream>

namespace CubbyDNN
{
class HiddenUnit : public ComputableUnit
{
public:
    //! Constructor
    //! \param inputTensorInfoVector : vector of TensorInfo
    //! \param outputTensorInfo : TensorInfo of the output m_tensor
    HiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
               TensorInfo outputTensorInfo);
    ~HiddenUnit() = default;

    HiddenUnit(const HiddenUnit& hiddenUnit) = delete;
    HiddenUnit(HiddenUnit&& hiddenUnit) noexcept;

    HiddenUnit& operator=(const HiddenUnit& hiddenUnit) = delete;
    HiddenUnit& operator=(HiddenUnit&& hiddenUnit) noexcept;

    //! Adds output computable unit ptr to ComputableUnit
    //! \param computableUnitPtr : ptr to output computable unit
    std::size_t AddOutputPtr(
        const SharedPtr<ComputableUnit>& computableUnitPtr);

    //! Adds input computable unit ptr to this ComputableUnit
    //! \param computableUnitPtr : ptr to input computable unit
    //! \param index : indicates order of input argument.
    void AddInputPtr(const SharedPtr<ComputableUnit>& computableUnitPtr,
                     std::size_t index);


    //! Determines whether system is ready to compute
    bool IsReady() final;

    //! Forward propagation
    virtual void Forward()
    {
    }

    //! Backward propagation
    virtual void Backward()
    {
    }

protected:
    std::unique_ptr<TensorOperation> m_tensorOperation = std::unique_ptr<
        NaiveOperation>();

private:
    /// ptr to units to receive result from
    std::vector<SharedPtr<ComputableUnit>> m_inputPtrVector;
    /// ptr to units to write result
    std::vector<SharedPtr<ComputableUnit>> m_outputPtrVector;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_HIDDENUNIT_HPP

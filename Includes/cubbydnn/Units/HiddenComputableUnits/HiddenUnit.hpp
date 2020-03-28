/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_HIDDENUNIT_HPP
#define CUBBYDNN_HIDDENUNIT_HPP

#include <cubbydnn/Computations/TensorOperations/NaiveOperations.hpp>
#include <cubbydnn/Computations/TensorOperations/TensorOperations.hpp>
#include <cubbydnn/Units/ComputableUnit.hpp>
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


    //! Forward propagation
    virtual void Forward() = 0;

    //! Backward propagation
    virtual void Backward() = 0;

protected:
    std::unique_ptr<TensorOperation> m_tensorOperation =
        std::unique_ptr<NaiveOperation>();
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_HIDDENUNIT_HPP

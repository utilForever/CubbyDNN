//
// Created by jwkim98 on 8/13/19.
//

#ifndef CUBBYDNN_HIDDENUNIT_HPP
#define CUBBYDNN_HIDDENUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
class HiddenUnit : public ComputableUnit
{
 public:
    //! Constructor
    //! \param inputTensorInfoVector : vector of TensorInfo
    //! \param outputTensorInfoVector : TensorInfo of the output m_tensor
    HiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
               std::vector<TensorInfo> outputTensorInfoVector);

    ~HiddenUnit() = default;

    HiddenUnit(const HiddenUnit& hiddenUnit) = delete;

    HiddenUnit& operator=(const HiddenUnit& hiddenUnit) = delete;

    //! Determines whether system is ready to compute
    bool IsReady() final;

    void Compute() override
    {
        // std::cout << "HiddenUnit" << std::endl;
        // std::cout << m_unitState.StateNum << std::endl;
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_HIDDENUNIT_HPP

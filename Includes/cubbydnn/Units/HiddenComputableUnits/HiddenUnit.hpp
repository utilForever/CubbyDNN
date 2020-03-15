/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_HIDDENUNIT_HPP
#define CUBBYDNN_HIDDENUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Computations/TensorOperations/TensorOperations.hpp>
#include <cubbydnn/Computations/TensorOperations/NaiveOperations.hpp>
#include <iostream>

namespace CubbyDNN
{
class HiddenUnit : public ComputableUnit
{
public:
    //! Constructor
    //! \param inputTensorInfoVector : vector of TensorInfo
    //! \param outputTensorInfo : TensorInfo of the output m_tensor
    //! \param numberOfOutputs : number of outputs that this unit is connected
    HiddenUnit(std::vector<TensorInfo> inputTensorInfoVector,
               TensorInfo outputTensorInfo, std::size_t numberOfOutputs = 1);
    ~HiddenUnit() = default;

    HiddenUnit(const HiddenUnit& hiddenUnit) = delete;
    HiddenUnit(HiddenUnit&& hiddenUnit) noexcept;

    HiddenUnit& operator=(const HiddenUnit& hiddenUnit) = delete;
    HiddenUnit& operator=(HiddenUnit&& hiddenUnit) noexcept;

    //! Determines whether system is ready to compute
    bool IsReady() final;

    void Compute() override
    {
        //std::cout << "hiddenUnit" << std::endl;
    }

protected:
    std::unique_ptr<TensorOperation> m_tensorOperation = std::unique_ptr<
        NaiveOperation>();
};

class MatMul : public HiddenUnit
{
public:
    MatMul(const TensorInfo& inputA, const TensorInfo& inputB,
           const TensorInfo& output, std::size_t numberOfOutputs);

    ~MatMul() = default;

    MatMul(const MatMul& matMul) = delete;

    MatMul& operator=(const MatMul& matMul) = delete;

    void Compute() override;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_HIDDENUNIT_HPP

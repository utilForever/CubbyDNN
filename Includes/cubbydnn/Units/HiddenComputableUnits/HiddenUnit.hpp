/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_HIDDENUNIT_HPP
#define CUBBYDNN_HIDDENUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>
#include <cubbydnn/Computations/Functions/Matrix.hpp>

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
    }
};

class MatMul : public HiddenUnit
{
public:
    MatMul(const TensorInfo& inputA, const TensorInfo& inputB,
           const TensorInfo& output);

    ~MatMul() = default;

    MatMul(const MatMul& matmul) = delete;

    MatMul& operator=(const MatMul& matmul) = delete;

    void Compute() override;
};
} // namespace CubbyDNN

#endif  // CUBBYDNN_HIDDENUNIT_HPP

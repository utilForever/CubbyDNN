// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTE_LOSSFUNCTIONS_HPP
#define CUBBYDNN_COMPUTE_LOSSFUNCTIONS_HPP

#include <cubbydnn/Tensors/Tensor.hpp>

namespace CubbyDNN::Compute
{
template <typename T>
class BaseLoss
{
public:

    BaseLoss() = default;
    virtual ~BaseLoss() = default;

    BaseLoss(const BaseLoss& loss) = default;
    BaseLoss(BaseLoss&& loss) noexcept = default;
    BaseLoss& operator=(const BaseLoss& loss) = default;
    BaseLoss& operator=(BaseLoss&& loss) noexcept = default;

    [[nodiscard]] virtual T Apply(Tensor& input, const Tensor& label) const = 0;

    virtual void ApplyDerivative(const Tensor& label, const Tensor& prevInput,
                                 Tensor& delta) const = 0;

protected:
    static void m_checkArguments(const Tensor& inputA, const Tensor& inputB)
    {
        const auto shape = inputA.TensorShape;
        const auto device = inputA.Device;

        if (inputA.TensorShape != inputB.TensorShape)
            throw std::invalid_argument(
                "BaseLoss - Tensor shape mismatch");

        if (inputA.NumericType != inputB.NumericType)
            throw std::invalid_argument("BaseLoss  - Numeric type mismatch");

        if (inputA.Device != inputB.Device)
            throw std::invalid_argument("BaseLoss - Device mismatch");
    }
};

template <typename T>
class MSE : public BaseLoss<T>
{
public:
    MSE()
        : BaseLoss<T>()
    {
    }

    ~MSE() override = default;

    MSE(const MSE& mse) = default;
    MSE(MSE&& mse) noexcept = default;
    MSE& operator=(const MSE& mse) = default;
    MSE& operator=(MSE&& mse) noexcept = default;

    [[nodiscard]] T Apply(Tensor& input, const Tensor& label) const override
    {
        BaseLoss<T>::m_checkArguments(input, label);

        const auto inputShape = input.TensorShape;
        const auto outputShape = label.TensorShape;

        const std::size_t colDataSizeInput = input.GetColumnElementSize();
        const std::size_t colDataSizeLabel = label.GetColumnElementSize();

        const auto numRows = inputShape.NumRows();
        const auto numCols = inputShape.NumCols();
        const auto batchSize = inputShape.NumMatrices();
        const auto matrixSizeInput = numRows * colDataSizeInput;
        const auto matrixSizeOutput = numRows * colDataSizeLabel;

        const T* inputPtr = static_cast<T*>(input.DataPtr);
        T* outputPtr = static_cast<T*>(label.DataPtr);

        T batchSum = 0;
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
            {
                T sum = 0;
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    auto temp = outputPtr[batchIdx * matrixSizeOutput +
                                          i * colDataSizeLabel + j] -
                                inputPtr[batchIdx * matrixSizeInput +
                                         i * colDataSizeInput + j];
                    sum += temp * temp;
                }
                batchSum += sum;
            }
        batchSum /= (2 * static_cast<T>(batchSize * numRows));

        return batchSum;
    }

    void ApplyDerivative(const Tensor& label, const Tensor& prevInput,
                         Tensor& delta) const override
    {
        BaseLoss<T>::m_checkArguments(prevInput, label);

        const auto deltaShape = delta.TensorShape;
        const auto labelShape = label.TensorShape;

        const std::size_t colDataSizeDelta = delta.GetColumnElementSize();
        const std::size_t colDataSizeLabel = label.GetColumnElementSize();
        const std::size_t colDataSizeInput = prevInput.GetColumnElementSize();

        const auto numRows = deltaShape.NumRows();
        const auto numCols = deltaShape.NumCols();
        const auto batchSize = deltaShape.NumMatrices();
        const auto matrixSizeInput = numRows * colDataSizeInput;
        const auto matrixSizeDelta = numRows * colDataSizeDelta;
        const auto matrixSizeLabel = numRows * colDataSizeLabel;

        T* deltaPtr = static_cast<T*>(delta.DataPtr);
        const T* prevInputPtr = static_cast<T*>(prevInput.DataPtr);
        const T* labelPtr = static_cast<T*>(label.DataPtr);

        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
            {
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    deltaPtr[batchIdx * matrixSizeDelta + i * colDataSizeDelta +
                             j] = labelPtr[batchIdx * matrixSizeLabel +
                                           i * colDataSizeLabel + j] -
                                  prevInputPtr[
                                      batchIdx * matrixSizeInput + i *
                                      colDataSizeInput +
                                      j];
                }
            }
    }
};
}
#endif

// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef TAKION_COMPUTE_LOSSFUNCTIONS_HPP
#define TAKION_COMPUTE_LOSSFUNCTIONS_HPP

#include <Takion/Tensors/Tensor.hpp>
#include <stdexcept>

namespace Takion::Compute
{
template <typename T>
class BaseLoss
{
public:

    BaseLoss() = default;
    virtual ~BaseLoss() = default;

    BaseLoss(const BaseLoss<T>& loss) = default;
    BaseLoss(BaseLoss<T>&& loss) noexcept = default;
    BaseLoss<T>& operator=(const BaseLoss<T>& loss) = default;
    BaseLoss<T>& operator=(BaseLoss<T>&& loss) noexcept = default;

    [[nodiscard]] virtual T Apply(const Tensor& input,
                                  const Tensor& label) const = 0;

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

        if (inputA.Device != inputB.Device)
            throw std::invalid_argument("BaseLoss - Device mismatch");

        if (inputA.BatchSize != inputB.BatchSize)
            throw std::invalid_argument(
                "Batch size mismatch between given inputs");
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

    [[nodiscard]] T
    Apply(const Tensor<T>& input, const Tensor<T>& label) const override
    {
        BaseLoss<T>::m_checkArguments(input, label);

        const auto inputShape = input.TensorShape;
        const auto outputShape = label.TensorShape;

        const std::size_t colDataSizeInput = input.GetColumnElementSize();
        const std::size_t colDataSizeLabel = label.GetColumnElementSize();

        const auto numRows = inputShape.NumRow();
        const auto numCols = inputShape.NumCol();
        const auto batchSize = input.BatchSize;
        const auto elementSize = input.ElementSize();
        const auto matrixSizeInput = numRows * colDataSizeInput;
        const auto matrixSizeOutput = numRows * colDataSizeLabel;

        const T* inputPtr = static_cast<T*>(input.DataPtr);
        const T* labelPtr = static_cast<T*>(label.DataPtr);

        T batchSum = 0;
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
            {
                T sum = 0;
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    const auto inputVal = inputPtr[batchIdx * matrixSizeInput +
                                                   i * colDataSizeInput + j];
                    const auto labelVal =
                        labelPtr[batchIdx * matrixSizeOutput +
                                 i * colDataSizeLabel + j];
                    auto temp = labelVal - inputVal;
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
                    const auto difference =
                        labelPtr[batchIdx * matrixSizeLabel +
                                 i * colDataSizeLabel + j] -
                        prevInputPtr[
                            batchIdx * matrixSizeInput + i *
                            colDataSizeInput +
                            j];

                    deltaPtr[batchIdx * matrixSizeDelta +
                             i * colDataSizeDelta + j] = difference;
                }
            }
    }
};
}
#endif

// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_COMPUTE_LOSSFUNCTIONS_HPP
#define CUBBYDNN_COMPUTE_LOSSFUNCTIONS_HPP

#include <cubbydnn/Tensors/Tensor.hpp>
#include <cmath>

namespace CubbyDNN::Compute
{
template <typename T>
class Loss
{
public:

    Loss() = default;
    virtual ~Loss() = default;

    Loss(const Loss& loss) = default;
    Loss(Loss&& loss) noexcept = default;
    Loss& operator=(const Loss& loss) = default;
    Loss& operator=(Loss&& loss) noexcept = default;

    [[nodiscard]] virtual T Apply(Tensor& input, const Tensor& label) const = 0;

    virtual void ApplyDerivative(const Tensor& label, const Tensor& prevInput,
                                 Tensor& delta) const = 0;

protected:
    static void m_checkArguments(std::vector<const Tensor&> arguments)
    {
        const auto shape = arguments.at(0).TensorShape;
        const auto numericType = arguments.at(0).NumericType;
        const auto device = arguments.at(0).Device;

        for (const auto& tensor : arguments)
        {
            if (tensor.TensorShape != shape)
                throw std::invalid_argument(
                    "Loss - Tensor shape mismatch");

            if (tensor.NumericType != numericType)
                throw std::invalid_argument("Loss  - Numeric type mismatch");

            if (tensor.Device != device)
                throw std::invalid_argument("Loss - Device mismatch");
        }
    }
};

template <typename T>
class MSE : public Loss<T>
{
public:
    MSE() = default;
    ~MSE() = default;

    MSE(const MSE& mse) = default;
    MSE(MSE&& mse) noexcept = default;
    MSE& operator=(const MSE& mse) = default;
    MSE& operator=(MSE&& mse) noexcept = default;

    [[nodiscard]] T Apply(Tensor& input, const Tensor& label) const override
    {
        Loss<T>::m_checkArguments({ input, label });

        const auto inputShape = input.TensorShape;
        const auto outputShape = label.TensorShape;

        const std::size_t colDataSizeInput = input.GetColumnElementSize();
        const std::size_t colDataSizeLabel = label.GetColumnElementSize();

        const auto numRows = inputShape.NumRows();
        const auto numCols = inputShape.NumCols();
        const auto batchSize = inputShape.NumMatrices();
        const auto matrixSizeInput = numRows * colDataSizeInput;
        const auto matrixSizeOutput = numRows * colDataSizeLabel;

        const T* inputPtr = static_cast<T*>(input.DataPtr.get());
        T* outputPtr = static_cast<T*>(label.DataPtr.get());

        T batchSum = 0;
        for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
            for (std::size_t i = 0; i < numRows; ++i)
            {
                T sum = 0;
                for (std::size_t j = 0; j < numCols; ++j)
                {
                    sum += std::pow(outputPtr[batchIdx * matrixSizeOutput +
                                              i * colDataSizeLabel + j] -
                                    inputPtr[batchIdx * matrixSizeInput +
                                             i * colDataSizeInput + j], 2);
                }
                batchSum += sum;
            }
        batchSum /= batchSize * numRows();

        return batchSum;
    }

    void ApplyDerivative(const Tensor& label, const Tensor& prevInput,
                         Tensor& delta) const override
    {
        Loss<T>::m_checkArguments({ delta, prevInput, label });

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

        const T* deltaPtr = static_cast<T*>(delta.DataPtr.get());
        const T* prevInputPtr = static_cast<T*>(prevInput.DataPtr.get());
        T* labelPtr = static_cast<T*>(label.DataPtr.get());

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

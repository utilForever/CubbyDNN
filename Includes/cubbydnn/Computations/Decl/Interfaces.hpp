//
// Created by jwkim98 on 6/19/19.
//
#include <cubbydnn/Tensors/Decl/TensorData.hpp>

#ifndef CUBBYDNN_INTERFACES_HPP
#define CUBBYDNN_INTERFACES_HPP
namespace CubbyDNN
{
template <typename T>
class ComputationUnit
{
 protected:
    std::vector<TensorDataPtr<T>> m_inputDataVector;
    std::vector<TensorDataPtr<T>> m_outputDataVector;

 public:
    /// start computation using input vector, and outputs its result to output
    /// vector

    ComputationUnit() = default;

    ComputationUnit(size_t inputSize, size_t outputSize)
        : m_inputDataVector(std::vector<TensorDataPtr<T>>(inputSize)),
          m_outputDataVector(std::vector<TensorDataPtr<T>>(outputSize))
    {
    }

    virtual void Compute() = 0;

    size_t GetInputSize()
    {
        return m_inputDataVector.size();
    }

    size_t GetOutputSize()
    {
        return m_outputDataVector.size();
    }

    /**
     * Sets Input before computation
     * @param tensorData : Data to set to the input
     * @param index : index of the input vector
     */
    void SetInput(TensorDataPtr<T>&& tensorData, size_t index)
    {
        // TODO : Error needs to be handled if index is invalid
        assert(index < m_inputDataVector.size());

        assert(m_outputDataVector.at(index) == nullptr);
        m_inputDataVector.at(index) = std::move(tensorData);
    }

    /**
     * Gets input that this operation has after process has been done
     * After receiving used input, dataPtr should be returned back
     * @param index : index of the input vector
     * @return : TensorDataPtr of the index
     */
    TensorDataPtr<T>&& GetInput(size_t index)
    {
        // TODO : Error needs to be handled if index is invalid
        assert(index < m_inputDataVector.size());

        assert(m_outputDataVector.at(index) != nullptr);
        TensorDataPtr<T>&& dataPtr = std::move(m_inputDataVector.at(index));
        m_inputDataVector.at(index) = nullptr;
        return dataPtr;
    }

    /**
     * Sets output to write data before computation
     * @param tensorData : tensor data to set on the output vector
     * @param index : index of the output vector
     */
    void SetOutput(TensorDataPtr<T>&& tensorData, size_t index)
    {
        // TODO : Error needs to be handled if index is invalid
        assert(index < m_outputDataVector.size());

        assert(m_outputDataVector.at(index) == nullptr);
        if (m_outputDataVector.at(index) == nullptr)
            m_outputDataVector.at(index) = std::move(tensorData);
    }

    /**
     * Gets output after the computation
     * @param index : index of output vector to receive output
     * @return : tensorDataPtr of the index
     */
    TensorDataPtr<T>&& GetOutput(size_t index)
    {
        // TODO : Error needs to be handled if index is invalid
        assert(index < m_outputDataVector.size());

        assert(m_outputDataVector.at(index) != nullptr);
        TensorDataPtr<T>&& dataPtr = std::move(m_outputDataVector.at(index));
        return dataPtr;
    }
};

template <typename T>
class ComputeAdd : public ComputationUnit<T>
{
    ComputeAdd() : ComputationUnit<T>(2, 1)
    {
    }

    void Compute() override
    {
        TensorDataPtr<T> a =
            std::move(ComputationUnit<T>::m_inputDataVector.at(0));
        TensorDataPtr<T> b =
            std::move(ComputationUnit<T>::m_inputDataVector.at(0));
        // m_outputDataVector.at(1);
        TensorDataPtr<T> result =
            std::move(ComputationUnit<T>::m_outputDataVector.at(0));

        assert(a->shape == b->shape);
        for (int i = 0; i < a->dataVec.size(); i++)
        {
            result->dataVec[i] = a->dataVec[i] + b->dataVec[i];
        }

        ComputationUnit<T>::m_inputDataVector.at(0) = std::move(a);
        ComputationUnit<T>::m_inputDataVector.at(1) = std::move(b);

        ComputationUnit<T>::m_outputDataVector.at(0) = std::move(result);
    }
};
}  // namespace CubbyDNN

#endif  // CUBBYDNN_INTERFACES_HPP

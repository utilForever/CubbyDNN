/// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_SOURCEUNIT_HPP
#define CUBBYDNN_SOURCEUNIT_HPP

#include <cubbydnn/Units/ComputableUnit.hpp>

namespace CubbyDNN
{
//! Unit that has no input, but has output only.
//! This type of unit must be able to fetch data(from the disk or cache)
//! or generator
class SourceUnit : public ComputableUnit
{
public:
    //! Constructor
    //! \param outputTensorInfoVector : TensorInfo of the output m_tensor(Which
    //! is always less than 1)
    explicit SourceUnit(std::vector<TensorInfo> outputTensorInfoVector);

    //! SourceUnit is not copy-assignable
    SourceUnit(const SourceUnit& sourceUnit) = delete;

    ~SourceUnit() = default;

    //! SourceUnit is not copy-assignable
    SourceUnit& operator=(const SourceUnit& sourceUnit) = delete;

    //! Checks if source is ready
    //! \return : true if ready to be computed false otherwise
    bool IsReady() final;

    void Compute() override
    {
    }
};

class ConstantUnit : public SourceUnit
{
public:
    explicit ConstantUnit(TensorInfo output, int numberOfOutputs,
                          void* dataPtr);

    //! ConstantUnit is not copy-assignable
    ConstantUnit(const ConstantUnit& sourceUnit) = delete;

    ~ConstantUnit()
    {
        free(m_dataPtr);
    }

    //! ConstantUnit is not copy-assignable
    SourceUnit& operator=(const SourceUnit& sourceUnit) = delete;

private:
    void* m_dataPtr = nullptr;
    size_t m_byteSize = 0;
};

//! Initializes constant with given data
template <typename T>
void InitializeConstant(TensorInfo tensorInfo, void* dataPtr,
                        const std::vector<std::vector<T>>& initializer)
{
    const auto shape = tensorInfo.GetShape();
    assert(initializer.size() > 0);
    assert(initializer.at(0).size() > 0);
    assert(initializer.size() == shape.Row);
    assert(initializer.at(0).size() == shape.Col);

    void* data = AllocateData<float>(tensorInfo.GetShape());

    for (size_t batchIdx = 0; batchIdx < shape.Batch; ++batchIdx)
        for (size_t channelIdx = 0; channelIdx < shape.Channel; ++channelIdx)
            for (size_t rowIdx = 0; rowIdx < shape.Row; ++rowIdx)
                for (size_t colIdx = 0; colIdx < shape.Col; ++colIdx)
                {
                    size_t offset = 0;
                    offset += colIdx;
                    size_t multiplier = shape.Col;
                    offset += multiplier * rowIdx;
                    multiplier *= shape.Row;
                    offset += multiplier * channelIdx;
                    multiplier *= shape.Channel;
                    offset += multiplier * batchIdx;
                    *(static_cast<T*>(dataPtr) + offset) = initializer
                                                           .at(rowIdx).at(
                                                               colIdx);
                }
}
} // namespace CubbyDNN

#endif  // CUBBYDNN_SOURCEUNIT_HPP

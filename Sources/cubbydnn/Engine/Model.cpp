// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <cubbydnn/Units/HiddenComputableUnits/Dense.hpp>
#include <cubbydnn/Engine/Model.hpp>
#include <cubbydnn/Units/SourceComputableUnits/SourceUnit.hpp>
#include <cubbydnn/Units/SinkComputableUnits/SinkUnit.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/Reshape.hpp>

namespace CubbyDNN::Graph
{
Model::Model(NumberSystem numberSystem)
    : m_numericType(numberSystem),
      m_unitManager(10)
{
}

void Model::Predict()
{
    m_unitManager.Predict();
}

void Model::Fit(std::size_t epochs)
{
    m_unitManager.Forward(epochs);
}

UnitId Model::PlaceHolder(const Shape& shape, const std::string& name)
{
    return m_unitManager.CreateSource<PlaceHolderUnit>(
        name, shape, m_numericType);
}

// TODO : put activation, initializing methods, etc.
UnitId Model::Dense(const UnitId& input, std::size_t units,
                    Activation activation,
                    std::unique_ptr<Initializer> kernelInitializer,
                    std::unique_ptr<Initializer> biasInitializer,
                    float dropoutProb, const std::string& name)
{
    if (dropoutProb < 0 || dropoutProb >= 1)
        throw std::invalid_argument("dropout rate must be between 0 and 1");

    Shape inputShape = m_unitManager.GetUnit(input)->GetOutputShape();

    const Shape weightShape = { inputShape.NumRows(), units };
    const Shape biasShape = { 1, inputShape.NumRows() };
    Shape outputShape = inputShape;
    outputShape.SetNumCols(units);

    return m_unitManager.CreateHidden<DenseUnit>(
        input, name, inputShape, weightShape, biasShape,
        outputShape, m_numericType, kernelInitializer, biasInitializer,
        activation, dropoutProb);
}

UnitId Model::Reshape(const UnitId& input, const Shape& shape,
                      const std::string& name)
{
    Shape inputShape = m_unitManager.GetUnit(input)->GetOutputShape();
    return m_unitManager.CreateHidden<ReshapeUnit>(input, name, inputShape,
                                                   shape, m_numericType);
}

void Model::Compile(const UnitId& outputUnitId, OptimizerType optimizer, Loss loss)
{
    UnitId sinkUnitId = { UnitBaseType::Sink, 0 };
    auto outputShape = m_unitManager.GetUnit(outputUnitId)->GetOutputShape();
    if (loss == Loss::CrossEntropy)
    {
        m_unitManager.CreateSink<CrossEntropy>(outputUnitId,
                                               "name", outputShape,
                                               m_numericType);
    }
    else
    {
        // TODO : implement other kinds of loss functions
        throw std::runtime_error("Not Implemented");
    }

    m_unitManager.CreateExecutionOrder();
    m_unitManager.AssignCopyUnits();
}
} // namespace CubbyDNN

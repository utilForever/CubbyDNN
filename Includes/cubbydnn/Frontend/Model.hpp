// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_MODEL_HPP
#define CUBBYDNN_MODEL_HPP

#include <cubbydnn/Utils/Declarations.hpp>
#include <cubbydnn/Engine/Engine.hpp>
#include <vector>

namespace CubbyDNN
{

//! Contains information required to build unit
struct Unit
{
    Unit(UnitType unitType);
    UnitType Type = UnitType::None;
    std::vector<TensorInfo> InputTensorInfoVector;
    TensorInfo OutputTensorInfo;
    std::vector<size_t> InputIdVector;
    size_t ID = 0;
    Initializer Init = Initializer::None;
    void* Data = nullptr;
};

class Model
{
public:
    Model(NumberSystem numberSystem = NumberSystem::Float32);

    Unit Constant(Shape shape, void* data);

    Unit Variable(Shape shape);

    Unit Variable(Shape shape, void* data);

    Unit PlaceHolder(Shape shape);

    Unit Input();

    Unit Add(const Unit& inputA, const Unit& inputB);

    Unit Mul(const Unit& inputA, const Unit& inputB);

    Unit Dense(const Unit& previous, size_t units, bool use_bias = true,
               Initializer kernelInitializer = Initializer::Xavier,
               Initializer biasInitializer = Initializer::Zeros);

    Unit Activate(const Unit& previous, Activation activation);

    Unit Dropout(Unit previous, float rate);

    Unit Regularize(Unit previous, Regularization regularization);;

    Unit Reshape(Unit previous, Shape shape);

    Unit Conv(Unit previous, size_t filters, Shape kernelSize, Shape strides,
                Padding padding, size_t dilationRate = 1, bool use_bias = true,
                Initializer kernelInitializer = Initializer::Xavier,
                Initializer biasInitializer = Initializer::Zeros);

    Unit MaxPooling(Shape poolSize, Shape strides, Padding padding);

    Unit AveragePooling(Shape poolSize, Shape strides, Padding padding);

    void Compile();

    void Fit(size_t epochs);

private:
    std::vector<Unit> m_unitVector;

   NumberSystem m_numberSystem;
};
} // namespace CubbyDNN

#endif

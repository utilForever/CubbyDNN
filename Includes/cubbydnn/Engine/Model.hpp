//! Copyright (c) 2019 Chris Ohk, Justin Kim

//! We are making my contributions/submissions to this project solely in our
//! personal capacity and are not conveying any rights to any intellectual
//! property of any third parties.

#ifndef CUBBYDNN_ENGINE_HPP
#define CUBBYDNN_ENGINE_HPP

#include <cubbydnn/Computations/Initializers/InitializerType.hpp>
#include <cubbydnn/Engine/UnitManager.hpp>

namespace CubbyDNN::Graph
{
//! Singleton class for maintaining threads that execute the program
class Model
{
public:
    Model(NumberSystem numberSystem);

    UnitId PlaceHolder(const Shape& shape,
                       const std::string& name = "placeHolder");
    //! \param input : unit ID of previous unit
    //! \param units : size of output perceptrons
    //! \param activation : type of activation to use
    //! \param kernelInitializer : initializer of the kernel
    //! \param biasInitializer : initializer of the bias
    //! \param dropoutProb : percentage of units to dropout
    //! \param name : name of this unit
    UnitId Dense(const UnitId& input, std::size_t units, Activation activation,
                 std::unique_ptr<Initializer> kernelInitializer,
                 std::unique_ptr<Initializer> biasInitializer,
                 float dropoutProb = 0.0, const std::string& name = "Dense");

    UnitId Reshape(const UnitId& input, const Shape& shape,
                   const std::string& name = "ReshapeUnit");

    //! OptimizerType, Loss function
    void Compile(const UnitId& outputUnitId, OptimizerType optimizer, Loss loss);

    //! Trains the graph with given optimizer and loss function
    void Fit(std::size_t epochs);

    void Predict();

    void Predict(void* input, void* output, int workers);

private:

    NumberSystem m_numericType;
    UnitManager m_unitManager;
};
} // namespace CubbyDNN

#endif  // CAPTAIN_THREADMANAGER_HPP

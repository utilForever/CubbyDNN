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
    Model(NumberSystem numericType);

    UnitId PlaceHolder(const Shape& shape,
                       const std::string& name, Compute::Device device);

    //! \param input : unit ID of previous unit
    //! \param units : size of output perceptrons
    //! \param weightInitializer : initializer of the kernel
    //! \param biasInitializer : initializer of the bias
    //! \param name : name of this unit
    //! \param device : device to execute this unit
    UnitId Dense(const UnitId& input, std::size_t units,
                 std::unique_ptr<Initializer> weightInitializer,
                 std::unique_ptr<Initializer> biasInitializer,
                 const std::string& name,
                 Compute::Device device);

    UnitId Dropout(const UnitId& input, float keepRate);

    UnitId Activation(const UnitId& input, const std::string& activationName,
                      const std::string& name, Compute::Device device);

    UnitId Reshape(const UnitId& input, const Shape& shape,
                   const std::string& name = "ReshapeUnit");

    //! OptimizerType, Loss function
    void Compile(const std::string& optimizer, ParameterPack optimizerParams);

    //! Trains the graph with given optimizer and loss function
    void Train(std::size_t epochs);

    void Predict();

    void Predict(void* input, void* output, int workers);

private:

    NumberSystem m_numericType;
    UnitManager m_unitManager;

    std::size_t m_id = 0;
};
} // namespace CubbyDNN

#endif  // CAPTAIN_THREADMANAGER_HPP

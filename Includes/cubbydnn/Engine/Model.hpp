//! Copyright (c) 2019 Chris Ohk, Justin Kim

//! We are making my contributions/submissions to this project solely in our
//! personal capacity and are not conveying any rights to any intellectual
//! property of any third parties.

#ifndef CUBBYDNN_ENGINE_HPP
#define CUBBYDNN_ENGINE_HPP

#include <cubbydnn/Computations/Initializers/InitializerType.hpp>
#include <cubbydnn/Engine/UnitManager.hpp>

#include <cubbydnn/Computations/LossFunctions/LossFunctions.hpp>

namespace CubbyDNN::Graph
{
//! Singleton class for maintaining threads that execute the program
class Model
{
public:
    Model(NumberSystem numericType);

    UnitId DataLoader(const Shape& shape,
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

    UnitId Loss(const UnitId& input, const UnitId& label, std::string lossType,
                const std::string& name, Compute::Device device);

    UnitId Loss(
        const UnitId& input, const UnitId& label, const std::string& name,
        Compute::Device device,
        std::unique_ptr<Compute::BaseLoss<float>> customLoss);

    UnitId Constant(Tensor tensor, const std::string& name);

    UnitId Multiply(const UnitId& inputA, const UnitId& inputB);

    UnitId Add(const UnitId& inputA, const UnitId& inputB);

    //! OptimizerType, BaseLoss function
    void Compile(const std::string& optimizer,
                 Parameter optimizerParams) noexcept;

    //! Trains the graph with given optimizer and loss function
    void Train(std::size_t epochs, bool async = false);

    void Predict();

    void Predict(void* input, void* output, int workers);

private:

    NumberSystem m_numericType;
    UnitManager m_unitManager;

    std::size_t m_id = 0;
};
} // namespace CubbyDNN

#endif  // CAPTAIN_THREADMANAGER_HPP

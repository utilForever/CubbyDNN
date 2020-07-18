// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "SimpleMnist.hpp"
#include <cubbydnn/Engine/Model.hpp>

namespace CubbyDNN::Test
{
void SimpleMnistTest()
{
    const Compute::Device device(0, Compute::DeviceType::Cpu, " myDevice");

    Graph::Model model(NumberSystem::Float);

    Tensor tensor({ 10, 1 }, device, std::vector<float>(10, 1));
    Tensor inputTensor({ 10, 1 }, device, std::vector<float>(10, 3));

    auto id = model.Constant(tensor, "input");
    auto labelId = model.Constant(inputTensor, "label");

    id = model.Dense(id, 10, std::make_unique<XavierNormal>(),
                     std::make_unique<HeNormal>(), "dense1", device);

    id = model.Activation(id, "ReLU", "act1", device);

    id = model.Loss(id, labelId, "MSE", "loss", device);

    model.Compile("SGD",
                  Graph::Parameter({}, { { "epsilon", 0.01f } }, {}));

    model.Train(100);
}
}

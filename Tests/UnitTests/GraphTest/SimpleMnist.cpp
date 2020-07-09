// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include "SimpleMnist.hpp"

#include <cubbydnn/Engine/Model.hpp>

namespace CubbyDNN
{
void SimpleMnistTest()
{
    const Compute::Device device(0, Compute::DeviceType::Cpu, " myDevice");
    Graph::Model model(NumberSystem::Float);
    auto id = model.PlaceHolder({ 100, 3, 3 }, "input", device);

   id = model.Dense(id, 10, std::make_unique<XavierNormal>(),
                std::make_unique<HeNormal>(), "dense1", device);

   id = model.Activation(id, "ReLU", "act1", device);

   //model.Compile(id, std::make_unique<Compute::SGD>(0.01), "MSE");

   model.Train(100);
}
}

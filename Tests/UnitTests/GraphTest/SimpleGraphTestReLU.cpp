// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Takion/FrontEnd/Model.hpp>
#include "SimpleGraphTest.hpp"

namespace Takion::Test
{
using namespace FrontEnd;

void SimpleGraphTest()
{
    Model<float> model(Compute::Device(0, Compute::DeviceType::CPU, "device0"),
                       10);

    auto tensor =
        model.Constant(Shape({ 200 }), std::vector<float>(2000, 10), "input");
    const auto label = model.Constant(Shape({ 3 }), std::vector<float>(30, 1),
                                      "label");

    tensor = model.Dense(tensor, 100);
    tensor = model.ReLU(tensor);
    tensor = model.Dense(tensor, 25);
    tensor = model.ReLU(tensor);
    tensor = model.Dense(tensor, 3);
    tensor = model.ReLU(tensor);
    model.MSE(tensor, label, "MseLoss");

    model.Compile("SGD", Parameter({}, { { "epsilon", 0.00001f } }, {}));
    model.Fit(10000);
}
} // namespace Takion::Test

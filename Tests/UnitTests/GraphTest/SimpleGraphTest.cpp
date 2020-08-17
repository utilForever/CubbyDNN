// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Takion/FrontEnd/Model.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "SimpleGraphTest.hpp"

namespace Takion::Test
{
using namespace FrontEnd;

template <typename T>
class GetMnistData
{
public:
    GetMnistData(const std::vector<std::vector<T>>& data);

    std::vector<T> operator()()
    {
    }

private:
    std::vector<std::vector<T>> m_data;
    std::size_t m_cycle = 0;
};

template <typename T>
class GetMnistLabel
{
public:
    GetMnistLabel(const std::vector<T>& label);

    std::vector<T> operator()()
    {
    }

private:
    std::vector<T> m_label;
    std::size_t m_cycle;
};

template <typename T>
std::pair<std::vector<T>, std::vector<std::vector<T>>> GetMnistDataSet(
    std::filesystem::path path)
{
    std::vector<T> label(60000);
    std::vector<T> data(60000);

    std::string line;
    std::stringstream ss;

    std::ifstream file(path);
    // Read first line
    std::getline(file, line);

    T val;

    while (std::getline(file, line))
    {
        std::stringstream stream;
        std::vector<T> colData(785);
        std::size_t index = 0;
        bool isLabel = true;
        while (stream >> val)
        {
            if (isLabel)
            {
                label.at(index) = val;
                isLabel = false;
            }
            else
                colData.at(index) = val;

            if (stream.peek() == ',')
                stream.ignore();

            index++;
        }
    }
}


void SimpleGraphTestReLU()
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
    model.Fit(5000);
}

void SimpleGraphTestSigmoid()
{
    Model<float> model(Compute::Device(0, Compute::DeviceType::CPU, "device0"),
                       10);

    auto tensor =
        model.Constant(Shape({ 200 }), std::vector<float>(2000, 10), "input");
    const auto label =
        model.Constant(Shape({ 3 }), std::vector<float>(30, 1), "label");

    tensor = model.Dense(tensor, 100);
    tensor = model.Sigmoid(tensor);
    tensor = model.Dense(tensor, 25);
    tensor = model.Sigmoid(tensor);
    tensor = model.Dense(tensor, 3);
    tensor = model.Sigmoid(tensor);
    model.MSE(tensor, label, "MseLoss");

    model.Compile("SGD", Parameter({}, { { "epsilon", 0.00001f } }, {}));
    model.Fit(5000);
}

void MnistTrainTest()
{
    std::filesystem::path filepath;

    auto [label, data] = GetMnistDataSet<float>(filepath);

    auto getData = GetMnistData<float>(data);
    auto getLabel = GetMnistLabel<float>(label);

    Model<float> model(Compute::Device(0, Compute::DeviceType::CPU, "device0"),
                       100);
    auto tensor = model.PlaceHolder(Shape({ 785 }), getData, "DataLoader");
    auto labelTensor =
        model.PlaceHolder(Shape({ 10 }), getLabel, "LabelLoader");
    tensor = model.Dense(tensor, 150);
    tensor = model.ReLU(tensor);
    tensor = model.Dense(tensor, 50);
    tensor = model.ReLU(tensor);
    tensor = model.Dense(tensor, 10);
    tensor = model.SoftMax(tensor);
    model.MSE(tensor, labelTensor, "MSELoss");

    model.Compile("SGD", Parameter({}, { { "epsilon", 0.0001f } }, {}));
    model.Fit(5000);
}
} // namespace Takion::Test

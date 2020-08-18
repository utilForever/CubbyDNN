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
    GetMnistData(const std::vector<std::vector<T>>& data)
        : m_data(data)
    {
    }

    std::vector<T> operator()()
    {
        return m_data.at(m_cycle++);
    }

private:
    std::vector<std::vector<T>> m_data;
    std::size_t m_cycle = 0;
};

template <typename T>
class GetMnistLabel
{
public:
    GetMnistLabel(const std::vector<std::size_t>& label)
        : m_label(label)
    {
    }

    std::vector<T> operator()()
    {
        std::vector<T> label(10, 0);
        std::size_t labelIdx = m_label.at(m_cycle);
        label.at(labelIdx) = static_cast<T>(1);
        m_cycle++;
        return label;
    }

private:
    std::vector<std::size_t> m_label;
    std::size_t m_cycle = 0;
};

template <typename T>
std::pair<std::vector<std::size_t>, std::vector<std::vector<T>>>
GetMnistDataSet(
    std::filesystem::path path)
{
    std::vector<std::size_t> label(60000);
    std::vector<std::vector<T>> data(60000);

    std::string line;
    std::stringstream ss;

    std::ifstream file(path);
    // Read first line
    std::getline(file, line);

    T val;

    std::size_t lineIdx = 0;
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
                label.at(lineIdx) = static_cast<std::size_t>(val);
                isLabel = false;
            }
            else
                colData.at(index) = val;

            if (stream.peek() == ',')
                stream.ignore();

            index++;
        }

        data.at(lineIdx) = std::move(colData);
    }

    return std::make_pair(label, data);
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

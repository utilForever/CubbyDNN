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
    GetMnistData(std::vector<std::vector<T>> data,
                 std::vector<std::vector<std::size_t>> randomIndices,
                 std::size_t batchSize)
        : m_data(std::move(data)),
          m_randomIndices(std::move(randomIndices)),
          m_batchSize(batchSize)
    {
    }

    std::vector<T> operator()()
    {
        const std::size_t elemSize = 785;
        std::vector<T> dataVector(m_batchSize * elemSize);
        auto indices = m_randomIndices.at(m_cycle);
        for (std::size_t i = 0; i < m_batchSize; ++i)
        {
            const auto idx = indices.at(i);
            const auto data = m_data.at(idx);
            for (std::size_t dataIdx = 0; dataIdx < 785; ++dataIdx)
            {
                dataVector.at(i * elemSize + dataIdx) =
                    data.at(dataIdx) / static_cast<T>(255);
            }
        }
        m_cycle++;
        return dataVector;
    }

private:
    std::vector<std::vector<T>> m_data;
    std::vector<std::vector<std::size_t>> m_randomIndices;
    std::size_t m_batchSize;
    std::size_t m_cycle = 0;
};

template <typename T>
class GetMnistLabel
{
public:
    GetMnistLabel(std::vector<T> label,
                  std::vector<std::vector<std::size_t>> randomIndices,
                  std::size_t batchSize)
        : m_label(std::move(label)),
          m_randomIndices(std::move(randomIndices)),
          m_batchSize(batchSize)
    {
    }

    std::vector<T> operator()()
    {
        const std::size_t numCategories = 10;
        std::vector<T> labelVector(m_batchSize * numCategories);
        auto indices = m_randomIndices.at(m_cycle);
        for (std::size_t i = 0; i < m_batchSize; ++i)
        {
            const auto idx = indices.at(i);
            std::vector<T> label(numCategories, 0);
            label.at(static_cast<std::size_t>(m_label.at(idx))) =
                static_cast<T>(1);
            for (std::size_t labelIdx = 0; labelIdx < numCategories; ++labelIdx)
                labelVector.at(i * numCategories + labelIdx) = label.at(
                    labelIdx);
        }
        m_cycle++;
        return labelVector;
    }

private:
    std::vector<T> m_label;
    std::vector<std::vector<std::size_t>> m_randomIndices;
    std::size_t m_batchSize;

    std::size_t m_cycle = 0;
};

template <typename T>
std::pair<std::vector<T>, std::vector<std::vector<T>>>
GetMnistDataSet(
    std::filesystem::path path)
{
    std::vector<T> label(60000);
    std::vector<std::vector<T>> data(60000);

    std::string line;
    std::stringstream ss;

    std::ifstream file(path);

    if (!file.is_open())
        throw std::runtime_error("Could not open file");

    // Read first line
    std::getline(file, line);

    T val;

    std::size_t lineIdx = 0;
    while (std::getline(file, line))
    {
        std::stringstream stream(line);
        std::vector<T> colData(785);
        std::size_t index = 0;
        bool isLabel = true;
        while (stream >> val)
        {
            if (isLabel)
            {
                label.at(lineIdx) = val;
                isLabel = false;
            }
            else
                colData.at(index) = val;

            if (stream.peek() == ',')
                stream.ignore();

            index++;
        }

        data.at(lineIdx) = std::move(colData);
        lineIdx++;
    }

    return std::make_pair(label, data);
}

std::vector<std::vector<std::size_t>> GetRandomSequence(std::size_t batchSize,
                                                        std::size_t epochs,
                                                        std::size_t lineLength)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<std::size_t> dist(0, lineLength - 1);

    std::vector<std::vector<std::size_t>> epochBatchIndices(epochs);

    for (std::size_t epochIdx = 0; epochIdx < epochs; ++epochIdx)
    {
        std::vector<std::size_t> batchIndices(batchSize);
        for (std::size_t i = 0; i < batchSize; ++i)
        {
            const auto val = dist(mt);
            batchIndices.at(i) = val;
        }
        epochBatchIndices.at(epochIdx) = batchIndices;
    }

    return epochBatchIndices;
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
    std::filesystem::path filepath =
        "C:\\Users\\user\\Desktop\\Files\\projects\\Takion\\Mnist\\27352_34877_"
        "bundle_archive\\mnist_train.csv";

    const std::size_t batchSize = 150;
    const std::size_t epochs = 80000;

    const auto [label, data] = GetMnistDataSet<float>(filepath);
    const auto randomIndices = GetRandomSequence(batchSize, epochs, 60000);

    auto getData = GetMnistData<float>(data, randomIndices, batchSize);
    auto getLabel = GetMnistLabel<float>(label, randomIndices, batchSize);

    Model<float> model(Compute::Device(0, Compute::DeviceType::CPU, "device0"),
                       batchSize);
    auto tensor = model.PlaceHolder(Shape({ 785 }), getData, "DataLoader");
    auto labelTensor =
        model.PlaceHolder(Shape({ 10 }), getLabel, "LabelLoader");
    tensor = model.Dense(tensor, 200);
    tensor = model.ReLU(tensor);
    tensor = model.Dense(tensor, 100);
    tensor = model.ReLU(tensor);
    tensor = model.Dense(tensor, 50);
    tensor = model.ReLU(tensor);
    tensor = model.Dense(tensor, 10);
    tensor = model.SoftMax(tensor);
    model.CrossEntropy(tensor, labelTensor, "CrossEntropy Loss");

    model.Compile("SGD", Parameter({}, { { "epsilon", 0.0002f } }, {}));
    model.Fit(epochs);
}
} // namespace Takion::Test

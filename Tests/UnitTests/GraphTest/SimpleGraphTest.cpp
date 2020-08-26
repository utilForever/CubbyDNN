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
class GetMnistData : public Util::Loader<T>
{
public:
    GetMnistData(Shape shape,
                 std::vector<std::vector<T>> data,
                 std::vector<std::vector<std::size_t>> randomIndices,
                 std::size_t batchSize)
        : Util::Loader<T>(shape, batchSize),
          m_data(std::move(data)),
          m_randomIndices(std::move(randomIndices)),
          m_batchSize(batchSize)
    {
    }

    std::vector<T> operator()() override
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
class GetMnistLabel : public Util::Loader<T>
{
public:
    GetMnistLabel(Shape shape, std::vector<T> label,
                  std::vector<std::vector<std::size_t>> randomIndices,
                  std::size_t batchSize)
        : Util::Loader<T>(shape, batchSize),
          m_label(std::move(label)),
          m_randomIndices(std::move(randomIndices)),
          m_batchSize(batchSize)
    {
    }

    std::vector<T> operator()() override
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
    std::filesystem::path path, std::size_t numLine)
{
    std::vector<T> label(numLine);
    std::vector<std::vector<T>> data(numLine);

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
                                                        std::size_t numLine)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<std::size_t> dist(0, numLine - 1);

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

    model.Compile("SGD", Parameter({}, { { "LearningRate", 0.00001f } }, {}));
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

    model.Compile("SGD", Parameter({}, { { "LearningRate", 0.00001f } }, {}));
    model.Fit(5000);
}

template <typename T>
float EvaluateAccuracy(const std::vector<T>& prediction,
                       const std::vector<T>& label, Shape labelShape,
                       std::size_t batchSize)
{
    const auto size = labelShape.Size();
    std::size_t match = 0;
    std::size_t mismatch = 0;
    for (std::size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx)
    {
        int argMaxPrediction = -1;
        int argMaxLabel = -1;
        T maxPredVal = std::numeric_limits<T>::min();
        T maxLabelVal = std::numeric_limits<T>::min();
        for (int i = 0; i < static_cast<int>(size); ++i)
        {
            const auto idx = size * batchIdx + i;
            const auto predVal = prediction.at(idx);
            const auto labelVal = label.at(idx);
            if (maxPredVal < predVal)
            {
                maxPredVal = predVal;
                argMaxPrediction = i;
            }
            if (maxLabelVal < labelVal)
            {
                maxLabelVal = labelVal;
                argMaxLabel = i;
            }
        }
        if (argMaxPrediction == argMaxLabel)
            match += 1;
        else
            mismatch += 1;
    }
    return static_cast<float>(match) / static_cast<float>(match + mismatch);
}

void MnistTrainTest()
{
    std::filesystem::path filepath =
        "C:\\Users\\user\\Desktop\\Files\\projects\\Takion\\Mnist\\27352_34877_"
        "bundle_archive\\mnist_train.csv";

    const std::size_t batchSize = 150;
    const std::size_t epochs = 20000;

    const auto [label, data] = GetMnistDataSet<float>(filepath, 60000);
    const auto randomIndices = GetRandomSequence(batchSize, epochs, 60000);

    Shape dataShape(Shape({ 785 }));
    Shape labelShape(Shape({ 10 }));

    auto dataLoader = std::make_unique<GetMnistData<float>>(
        dataShape, data, randomIndices,
        batchSize);
    auto labelLoader = std::make_unique<GetMnistLabel<float>>(
        labelShape, label, randomIndices,
        batchSize);

    Model<float> model(Compute::Device(0, Compute::DeviceType::CPU, "device0"),
                       batchSize);

    auto tensor = model.Fetcher(dataShape, std::move(dataLoader), "DataLoader");
    auto labelTensor = model.Fetcher(labelShape, std::move(labelLoader),
                                     "LabelLoader");
    tensor = model.Dense(tensor, 200);
    tensor = model.ReLU(tensor);
    tensor = model.Dense(tensor, 100);
    tensor = model.ReLU(tensor);
    tensor = model.Dense(tensor, 50);
    tensor = model.ReLU(tensor);
    tensor = model.Dense(tensor, 10);
    tensor = model.SoftMax(tensor);
    model.CrossEntropy(tensor, labelTensor, "CrossEntropy Loss");

    model.Compile("SGD", Parameter({}, { { "LearningRate", 0.001f } }, {}));
    model.Fit(epochs);
}

void MnistTrainTest2()
{
#ifdef _MSC_VER
    std::filesystem::path trainFilePath =
        "C:\\Users\\user\\Desktop\\Files\\projects\\Takion\\Mnist\\27352_34877_"
        "bundle_archive\\mnist_train.csv";
    std::filesystem::path validationFilePath =
        "C:\\Users\\user\\Desktop\\Files\\projects\\Takion\\Mnist\\27352_34877_"
        "bundle_archive\\mnist_test.csv";
#else
    std::filesystem::path trainFilePath =
        "/mnt/c/Users/user/Desktop/Files/projects/Takion/Mnist/27352_34877_bundle_archive/mnist_train.csv";
    std::filesystem::path validationFilePath =
        "/mnt/c/Users/user/Desktop/Files/projects/Takion/Mnist/27352_34877_bundle_archive/mnist_test.csv";
#endif

    std::cout << "Train filepath : " << trainFilePath << std::endl;
    std::cout << "Validation filePath : " << validationFilePath << std::endl;

    // 모델 학습에 필요한 Hyperparameter 설정
    const std::size_t batchSize = 150;
    const std::size_t epochs = 20000;
    const std::size_t trainDataNumLine = 60000;
    const std::size_t validationDataNumLine = 10000;

    const auto [label, data] = GetMnistDataSet<float>(
        trainFilePath, trainDataNumLine);
    const auto [validationLabel, validationData] =
        GetMnistDataSet<float>(validationFilePath, validationDataNumLine);
    const auto trainRandomIndices = GetRandomSequence(
        batchSize, epochs, trainDataNumLine);
    const auto validationRandomIndices =
        GetRandomSequence(batchSize, epochs, validationDataNumLine);

    Shape dataShape(Shape({ 785 }));
    Shape labelShape(Shape({ 10 }));

    // Data loading 을 위한 클래스 (사용자가 직접 구현 가능)
    auto trainDataLoader = GetMnistData<float>(dataShape, data,
                                               trainRandomIndices,
                                               batchSize);
    auto trainLabelLoader = GetMnistLabel<float>(labelShape, label,
                                                 trainRandomIndices,
                                                 batchSize);

    auto validationDataLoader = GetMnistData<float>(
        dataShape, validationData, validationRandomIndices, batchSize);
    auto validationLabelLoader = GetMnistLabel<float>(
        dataShape, validationLabel, validationRandomIndices, batchSize);

    // Model 선언 (Device, batchSize) Device 를 통해 원하는 실행 방식을 설정 후 배치 학습에 사용할 배치의 크기 설정 
    Model<float> model(Compute::Device(0, Compute::DeviceType::CPU, "device0"),
                       batchSize);

    // Fetcher : 데이터를 로드하여 모델에 공급해주는 유닛
    auto dataFetcher = model.Fetcher(dataShape, "DataLoader");
    auto labelFetcher = model.Fetcher(labelShape, "LabelLoader");
    // Dense : Dense 연산 수행 유닛
    auto unit = model.Dense(dataFetcher, 200);
    // ReLU : ReLU Activation function 유닛
    unit = model.ReLU(unit);
    unit = model.Dense(unit, 100);
    unit = model.ReLU(unit);
    unit = model.Dense(unit, 50);
    unit = model.ReLU(unit);
    unit = model.Dense(unit, 10);
    // Softmax : Softmax Activation function 유닛 - 주로 다클래스 분류에 사용
    auto softMax = model.SoftMax(unit);
    // CrossEntropy : 다클래스 분류에 주로 사용되는 오차함수
    auto lossId = model.
        CrossEntropy(softMax, labelFetcher, "CrossEntropy Loss");

    // Compile 을 통해 지금까지 모델에 넣은 유닛들을 조합하여 실행 가능한 그래프 생성
    // 모델에 문제가 있을 경우 오류 메세지를 출력
    // Optimizer 는 SGD (Stochatic gradient descent) 를 사용하며 Learning Rate 는 0.005로 설정
    model.Compile("SGD", Parameter({}, { { "LearningRate", 0.0005f } }, {}));

    // Training 을 epoch 만큼 반복
    for (std::size_t cycle = 0; cycle < epochs; ++cycle)
    {
        // dataFetcher (data Fetcher 유닛의 리턴 id) 에 로더를 통해 학습에 필요한 데이터 공급
        // labelFetcher (label Fetcher 유닛의 리턴 id) 에 로더를 통해 학습에 필요한 데이터 공급
        model.Train({ { dataFetcher, trainDataLoader() } }, labelFetcher,
                    trainLabelLoader());

        if (cycle % 100 == 0)
        {
            // 100 cycle 주기로 validation data 를 통해 모델 평가
            auto validData = validationDataLoader();
            const auto validLabel = validationLabelLoader();
            // Predict : 학습을 진행하지 않고 data 와 label 을 넣어 현재 모델의 출력과 오차함수 계산
            model.Predict({ { dataFetcher, validData } }, labelFetcher,
                          validLabel);
            // Output : softMax 유닛의 출력을 가져옴 
            const auto outputData = model.Output(softMax);
            // SoftMax 의 출력값과 레이블을 비교하여 정확도 계산
            const auto accuracy = EvaluateAccuracy(outputData.Data, validLabel,
                                                   outputData.TensorShape,
                                                   outputData.BatchSize);
            // 오차함수 (CrossEntropy) 유닛의 loss 출력
            const auto loss = model.GetLoss(lossId);
            // 결과를 터미널 창에 출력
            std::cout << "epoch : " << cycle << " validation Loss : " << loss <<
                " validation Accuracy : " << accuracy * 100 << "%" << std::endl;
        }
    }
}
} // namespace Takion::Test

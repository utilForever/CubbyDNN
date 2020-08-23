#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Core/Shape.hpp>
#include <CubbyDNN/Datas/DataLoader.hpp>
#include <CubbyDNN/Datas/Dataset/MNISTDataset.hpp>
#include <CubbyDNN/Datas/TransformedDataset.hpp>
#include <CubbyDNN/Node/Parameter.hpp>
#include <CubbyDNN/Optimizer/Momentum.hpp>
#include <CubbyDNN/Preprocess/ImageTransforms.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

using namespace CubbyDNN;

auto main() -> int
{
    Core::Graph graph;

    auto x = graph.Builder().Input("x");
    auto y = graph.Builder().Input("y");

    auto w1 = graph.Builder().Parameter(
        "w1", Core::Shape{ 300, 784 }, graph.Builder().InitXavier(0, 784, 300));
    auto b1 = graph.Builder().Parameter("b1", Core::Shape{ 300 },
                                        graph.Builder().InitConstant());
    auto a1 = graph.Builder().Dense(x, w1, b1);
    auto o1 = graph.Builder().ReLU(a1, .001f);

    auto w2 = graph.Builder().Parameter("w2", Core::Shape{ 10, 300 },
                                        graph.Builder().InitXavier(0, 300, 10));
    auto b2 = graph.Builder().Parameter("b2", Core::Shape{ 10 },
                                        graph.Builder().InitConstant());
    auto a2 = graph.Builder().Dense(o1, w2, b2);
    auto o2 = graph.Builder().Softmax(a2, { true, false });

    auto loss = graph.Builder().SoftmaxCE(y, o2);

    Optimizer::Momentum optimizer(0.9f, { graph.Node<Node::Parameter>("w1"),
                                          graph.Node<Node::Parameter>("b1"),
                                          graph.Node<Node::Parameter>("w2"),
                                          graph.Node<Node::Parameter>("b2") });

    std::mt19937_64 engine(std::random_device{}());
    std::vector<std::size_t> shuffledIndexList;

    for (std::size_t index = 0; index < 60000; ++index)
    {
        shuffledIndexList.emplace_back(index);
    }

    auto trainDataset = MNISTDataset("./data", true, true)
                            .Transform(Transforms::ImageTransforms::ToMemory());
    auto testDataset = MNISTDataset("./data", false, true)
                           .Transform(Transforms::ImageTransforms::ToMemory());

    auto trainLoader = DataLoader(std::move(trainDataset), 32, true);
    auto testLoader = DataLoader(std::move(testDataset), 10000, false);

    while (true)
    {
        auto begin(std::chrono::system_clock::now());

        float runningLoss = 0;

        trainLoader.Begin();
        std::optional<SimpleBatch> batch;
        while ((batch = trainLoader.Next()).has_value())
        {
            graph.Feed(
                { { "x", Core::Shape{ 784, 32 }, batch.value().Data.GetSpan() },
                  { "y", Core::Shape{ 10, 32 },
                    batch.value().Target.GetSpan() } });

            runningLoss += loss.EvalOutput().Output()[0];

            auto g = o1.EvalOutput().Output();
            optimizer.Reduce(1e-3f, loss);
        }

        std::cout << "Training Loss: " << runningLoss  / trainLoader.GetSize()
                  << std::endl;

        testLoader.Begin();
        batch = testLoader.Next();

        graph.Feed(
            { { "x", Core::Shape{ 784, 10000 }, batch.value().Data.GetSpan() },
              { "y", Core::Shape{ 10, 10000 }, batch.value().Target.GetSpan() } });

        std::cout << "Test Loss: " << loss.EvalOutput().Output()[0]
                  << std::endl;

        auto end(std::chrono::system_clock::now());

        std::cout << "==== Time took : "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - begin)
                         .count()
                  << "ms ====" << std::endl
                  << std::endl;
    }

    return 0;
}
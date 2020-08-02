#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Core/Shape.hpp>
#include <CubbyDNN/Node/Parameter.hpp>
#include <CubbyDNN/Optimizer/Momentum.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

using namespace CubbyDNN;

auto main() -> int
{
    std::vector<float> trainX(60000 * 784);
    std::vector<float> trainY(60000 * 10);
    std::vector<float> testX(10000 * 784);
    std::vector<float> testY(10000 * 10);

    {
        std::ifstream input{ L"train_input.dat",
                             std::ifstream::binary | std::ifstream::in };
        input.read(reinterpret_cast<char*>(trainX.data()),
                   sizeof(float) * trainX.size());
    }

    {
        std::ifstream input{ L"train_label.dat",
                             std::ifstream::binary | std::ifstream::in };
        input.read(reinterpret_cast<char*>(trainY.data()),
                   sizeof(float) * trainY.size());
    }

    {
        std::ifstream input{ L"test_input.dat",
                             std::ifstream::binary | std::ifstream::in };
        input.read(reinterpret_cast<char*>(testX.data()),
                   sizeof(float) * testX.size());
    }

    {
        std::ifstream input{ L"test_label.dat",
                             std::ifstream::binary | std::ifstream::in };
        input.read(reinterpret_cast<char*>(testY.data()),
                   sizeof(float) * testY.size());
    }

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

    while (true)
    {
        auto begin(std::chrono::system_clock::now());

        graph.Feed({ { "x", Core::Shape{ 784, 60000 },
                       Core::Span<float>{ trainX.begin(), trainX.end() } },
                     { "y", Core::Shape{ 10, 60000 },
                       Core::Span<float>{ trainY.begin(), trainY.end() } } });

        std::cout << "Training Loss: " << loss.EvalOutput().Output()[0]
                  << std::endl;

        graph.Feed({ { "x", Core::Shape{ 784, 10000 },
                       Core::Span<float>{ testX.begin(), testX.end() } },
                     { "y", Core::Shape{ 10, 10000 },
                       Core::Span<float>{ testY.begin(), testY.end() } } });

        std::cout << "Test Loss: " << loss.EvalOutput().Output()[0]
                  << std::endl;

        for (std::size_t index = 1; index < 60000; ++index)
        {
            std::swap(
                shuffledIndexList[index - 1],
                shuffledIndexList[std::uniform_int_distribution<std::size_t>{
                    index, 60000 - 1 }(engine)]);
        }

        for (std::size_t index = 0; index + 32 <= 60000; index += 32)
        {
            std::size_t actualBatchSize =
                std::min<std::size_t>(60000 - index, 32);

            graph.Feed({ { "x", Core::Shape{ 784, actualBatchSize },
                           Core::Span<float>{ trainX.data() + index * 784,
                                              actualBatchSize * 784 } },
                         { "y", Core::Shape{ 10, actualBatchSize },
                           Core::Span<float>{ trainY.data() + index * 10,
                                              actualBatchSize * 10 } } });

            auto g = o1.EvalOutput().Output();

            optimizer.Reduce(0.001f, loss);
        }

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
#include <CubbyDNN/Core/Graph.hpp>
#include <CubbyDNN/Core/Shape.hpp>

#include <fstream>
#include <vector>

using namespace CubbyDNN;

auto main() -> int
{
    std::vector<float> trainX(60000 * 784);
    std::vector<float> trainY(60000 * 10);
    std::vector<float> testX(10000 * 784);
    std::vector<float> testY(10000 * 10);

    {
        std::ifstream sInput{ L"train_input.dat",
                              std::ifstream::binary | std::ifstream::in };
        sInput.read(reinterpret_cast<char *>(trainX.data()),
                    sizeof(float) * trainX.size());
    }

    {
        std::ifstream sInput{ L"train_label.dat",
                              std::ifstream::binary | std::ifstream::in };
        sInput.read(reinterpret_cast<char *>(trainY.data()),
                    sizeof(float) * trainY.size());
    }

    {
        std::ifstream sInput{ L"test_input.dat",
                              std::ifstream::binary | std::ifstream::in };
        sInput.read(reinterpret_cast<char *>(testX.data()),
                    sizeof(float) * testX.size());
    }

    {
        std::ifstream sInput{ L"test_label.dat",
                              std::ifstream::binary | std::ifstream::in };
        sInput.read(reinterpret_cast<char *>(testY.data()),
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

    return 0;
}
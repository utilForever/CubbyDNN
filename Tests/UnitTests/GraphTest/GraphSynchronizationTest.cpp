//
// Created by jwkim98 on 4/19/19.
//

#include "GraphSynchronizationTest.hpp"

namespace GraphTest
{
void SyncTest()
{
    std::vector<TensorData<int>> dataPool;
    auto shape = TensorShape(1,10,10);
    auto data = std::vector<int>(100);
    auto tensorData = TensorData<int>(data , shape);

    dataPool = std::vector(
            10, tensorData);

    auto Sender = Operation<int>();
    auto Receiver = Operation<int>();
}

//void Connect(Operation<int> op1, Operation<int> op2){
//    TensorDataPtr<int> = std::make_unique()
//}
}  // namespace GraphTest
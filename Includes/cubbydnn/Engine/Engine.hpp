//! Copyright (c) 2019 Chris Ohk, Justin Kim

//! We are making my contributions/submissions to this project solely in our
//! personal capacity and are not conveying any rights to any intellectual
//! property of any third parties.

#ifndef CUBBYDNN_ENGINE_HPP
#define CUBBYDNN_ENGINE_HPP

#include <cubbydnn/Engine/TaskWrapper.hpp>
#include <cubbydnn/Units/CopyUnit.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>
#include <cubbydnn/Units/SinkComputableUnits/SinkUnit.hpp>
#include <cubbydnn/Units/SourceComputableUnits/SourceUnit.hpp>
#include <cubbydnn/Utils/SharedPtr.hpp>
#include <cubbydnn/Utils/SpinLockQueue.hpp>
#include <vector>

namespace CubbyDNN
{
//! Singleton class for maintaining threads that execute the program
class Graph
{
public:
    Graph(NumberSystem numberSystem);

    //! Execute the graph using single thread
    void ExecuteForward(std::size_t epochs);
    //! Trains the graph with given optimizer and loss function
    void Fit(std::size_t epochs);

    UnitId PlaceHolder(const Shape& shape);


    UnitId Dense(const UnitId& input, std::size_t units,
                 ActivationType activationType);

    //! Optimizer, Loss function
    void Compile(Optimizer optimizer, Loss loss);

private:
    void m_getExecutionOrder(UnitId subjectUnit,
                             std::vector<std::vector<UnitId>>& executionOrder,
                             int depth);

    /// True if this engine is active false otherwise
    bool m_active = true;
    std::vector<std::vector<UnitId>> m_executionOrder;

    std::vector<SharedPtr<SourceUnit>> m_sourceUnitVector;
    SharedPtr<SinkUnit> m_sinkUnit;
    std::vector<SharedPtr<HiddenUnit>> m_hiddenUnitVector;
    std::vector<SharedPtr<CopyUnit>> m_sourceCopyUnitVector;
    std::vector<SharedPtr<CopyUnit>> m_hiddenCopyUnitVector;

    /// number of epochs to run the graph
    /// If stateNum reaches this, that unit will be no longer computed
    std::size_t m_maxEpochs = 0;
    std::atomic_bool m_ready = false;
    NumberSystem m_numberSystem;
};
} // namespace CubbyDNN

#endif  // CAPTAIN_THREADMANAGER_HPP

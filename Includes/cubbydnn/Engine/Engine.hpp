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

#include <list>
#include <thread>
#include <vector>

namespace CubbyDNN
{
//! Singleton class for maintaining threads that execute the program
class Graph
{
 protected:
    //! Enqueue tasks into task queue
    //! \param task
    void EnqueueTask(TaskWrapper&& task);

    //! Dequeue tasks from task queue
    //! \return
    TaskWrapper DequeueTask();

    //! Brings back size of the task queue
    //! \return : size of task queue
    std::size_t TaskQueueSize()
    {
        return m_taskQueue.Size();
    }

 public:
    Graph(NumberSystem numberSystem);

    //! Execute the graph using single thread
    void ExecuteForward(size_t epochs);

    //! Execute the graph backwards using single thread
    void ExecuteBackward(size_t epochs);

    //! Execute the graph using multiple workers
    void ExecuteForwardParallel(size_t workers, size_t epochs);

    //! Joins threads in the thread queue and terminates the execution
    void JoinThreads();

    //! Aborts all threads strictly
    void Abort();

    UnitIdentifier PlaceHolder(const Shape& shape);

    //! Adds hiddenUnit to HiddenUnitVector
    //! \param previousUnitVector : vector of previous units
    //! \param outputTensorInfo : vector of TensorInfo of outputs
    //! \param numberOfOutputs : number of outputs from this unit
    //! \return : assigned id of the unit
    UnitIdentifier Hidden(const std::vector<UnitIdentifier>& previousUnitVector,
                          TensorInfo outputTensorInfo,
                          size_t numberOfOutputs = 1);

    UnitIdentifier Dense(const UnitIdentifier& input, std::size_t units);

    //! Adds MultiplyUnit to HiddenUnitVector and assigns ID
    //! MultiplyUnit performs multiplications between two tensors (matrices) C=
    //! A*B \param unitA : first input operand information \param unitB : second
    //! input operand information
    UnitIdentifier Multiply(const UnitIdentifier& unitA,
                            const UnitIdentifier& unitB);

    //! Optimizer, Loss function
    void Compile();

 private:
    void m_getExecutionOrder(UnitIdentifier subjectUnit,
                             std::list<UnitIdentifier>& executionOrder)
    {
        const UnitType type = subjectUnit.Type;
        const std::size_t id = subjectUnit.ID;

        if (type == UnitType::Hidden)
        {
            const auto& hiddenUnit = m_hiddenUnitVector.at(id);
            const auto& inputUnitIdVector = hiddenUnit->GetInputUnitVector();
            executionOrder.emplace_front(hiddenUnit->GetIdentifier());
            for (const auto& unit : inputUnitIdVector)
            {
                m_getExecutionOrder(unit, executionOrder);
            }
        }
        else if (type == UnitType::Source)
        {
            const auto& sourceUnit = m_sourceUnitVector.at(id);
            executionOrder.emplace_front(sourceUnit->GetIdentifier());
        }
    }

    //! Connects between sourceUnit and intermediateUnit by assigning copyUnit
    //! between them
    //! \param originID : sourceUnit ID to connect
    //! \param destID : intermediateUnit ID of destination
    //! \param destInputIndex : input index of this connection to destination
    void m_connectSourceToHidden(size_t originID, size_t destID,
                                 size_t destInputIndex = 0);

    //! Connects between intermediateUnit and intermediateUnit by assigning
    //! copyUnit between them
    //! \param originID : unique ID of origin intermediateUnit
    //! \param destID : unique ID of destination intermediateUnit
    //! \param destInputIndex : input index of this connection to destination
    void m_connectHiddenToHidden(size_t originID, size_t destID,
                                 size_t destInputIndex = 0);

    //! Connects between intermediateUnit and sinkUnit by assigning
    //! copyUnit between them
    //! \param originID : unique ID of origin intermediateUnit
    //! \param destID : unique ID of destination sinkUnit
    //! \param destInputIndex : input index of this connection to destination
    void m_connectHiddenToSink(size_t originID, size_t destID,
                               size_t destInputIndex = 0);

    //! Connects unit with previous units using ID
    void m_connectWithPreviousUnit(
        const std::vector<UnitIdentifier>& previousUnitVector,
        UnitIdentifier subjectUnitIdentifier);

    //! Routine for thread which executes mainUnits
    void m_run();

    void m_executeForwardUnits();

    void m_executeCopyUnits();

    bool m_IsComplete(std::size_t epochs);

    std::thread m_scanMainThread;

    std::thread m_scanCopyThread;
    /// Stores active threads assigned for executing mainTasks
    std::vector<std::thread> m_mainThreadPool;
    /// Stores active threads assigned for executing copyTasks
    std::vector<std::thread> m_copyThreadPool;
    /// TaskQueue storing tasks from SourceUnit, IntermediateUnit and SinkUnit
    SpinLockQueue<TaskWrapper> m_taskQueue;
    /// True if this engine is active false otherwise
    bool m_active = true;
    std::list<UnitIdentifier> m_executionOrder;

    // TODO : find way to specify execution orderings
    std::vector<SharedPtr<SourceUnit>> m_sourceUnitVector;
    SharedPtr<SinkUnit> m_sinkUnit;
    std::vector<SharedPtr<HiddenUnit>> m_hiddenUnitVector;
    std::vector<SharedPtr<CopyUnit>> m_copyUnitVector;

    /// number of epochs to run the graph
    /// If stateNum reaches this, that unit will be no longer computed
    std::size_t m_maxEpochs = 0;
    std::atomic_bool m_ready = false;
    NumberSystem m_numberSystem;
};
}  // namespace CubbyDNN

#endif  // CAPTAIN_THREADMANAGER_HPP

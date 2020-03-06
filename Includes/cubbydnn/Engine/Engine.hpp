//! Copyright (c) 2019 Chris Ohk, Justin Kim

//! We are making my contributions/submissions to this project solely in our
//! personal capacity and are not conveying any rights to any intellectual
//! property of any third parties.

#ifndef CUBBYDNN_ENGINE_HPP
#define CUBBYDNN_ENGINE_HPP

#include <cubbydnn/Utils/SpinLockQueue.hpp>
#include <cubbydnn/Units/CopyUnit.hpp>
#include <cubbydnn/Units/HiddenComputableUnits/HiddenUnit.hpp>
#include <cubbydnn/Units/SinkComputableUnits/SinkUnit.hpp>
#include <cubbydnn/Units/SourceComputableUnits/SourceUnit.hpp>
#include <cubbydnn/Utils/SharedPtr.hpp>
#include <cubbydnn/Engine/TaskWrapper.hpp>

#include <functional>
#include <thread>
#include <vector>

namespace CubbyDNN
{


//! Singleton class for maintaining threads that execute the program
class Engine
{
protected:
    //! Enqueues tasks into task queue
    //! \param task
    static void EnqueueTask(TaskWrapper&& task);

    //! Dequeue tasks from task queue
    //! \return
    static TaskWrapper DequeueTask();

    //! Brings back size of the task queue
    //! \return : size of task queue
    static size_t TaskQueueSize()
    {
        return m_taskQueue.Size();
    }

public:

    //! Execute the graph using single thread
    static void Execute(size_t epochs);

    //! Execute the graph using multiple workers
    static void ExecuteParallel(size_t workers, size_t epochs);

    //! Joins threads in the thread queue and terminates the execution
    static void JoinThreads();

    static void Abort();

    //! Adds sourceUnit to sourceUnitVector and assigns ID for the unit
    //! \param outputTensorInfo:  vector of TensorInfo of outputs
    //! \return : assigned id of the unit
    static UnitIdentifier Source(
        const TensorInfo& outputTensorInfo, size_t numberOfOutputs = 1);

    //! Adds constant to sourceUnitVector and assigns ID for the unit
    //! \param output : output tensor information of this constant
    //! \param numberOfOutputs : number of output tensors
    //! \param dataPtr : ptr to data to set as constant this will be freed by destructor of ConstantUnit unit
    static UnitIdentifier Constant(const TensorInfo& output, void* dataPtr,
                                   int numberOfOutputs = 1);

    //! Adds hiddenUnit to HiddenUnitVector
    //! \param previousUnitVector : vector of previous units
    //! \param inputTensorInfoVector : vector of TensorInfo of inputs
    //! \param outputTensorInfo : vector of TensorInfo of outputs
    //! \return : assigned id of the unit
    static UnitIdentifier Hidden(
        const std::vector<UnitIdentifier>& previousUnitVector,
        TensorInfo outputTensorInfo, size_t numberOfOutputs = 1);

    //! Adds MultiplyUnit to HiddenUnitVector and assigns ID
    //! MultiplyUnit performs multiplications between two tensors (matrices) C= A*B
    //! \param inputA : first input operand information
    //! \param inputB : second input operand information
    //! \param output : output information
    static UnitIdentifier Multiply(
        const
        UnitIdentifier& unitA, const UnitIdentifier& unitB,
        size_t numberOfOutputs = 1);


    //! Adds sinkUnit to intermediateUnitVector and assigns ID for the unit
    //! \param inputTensorInfoVector : vector of TensorInfo of inputs
    //!  \return : assigned id of the unit
    static void Sink(
        const std::vector<UnitIdentifier>& previousUnit,
        const std::vector<TensorInfo>& inputTensorInfoVector);

    //! Adds sinkUnit to intermediateUnitVector and assigns ID for the unit
    //! \param inputTensorInfoVector : vector of TensorInfo of inputs
    //! \param testFunction : lambda used for testing the output
    //!  \return : assigned id of the unit
    static UnitIdentifier OutputTest(
        const UnitIdentifier& previousUnit,
        const std::function<void(const Tensor&, size_t)>&
        testFunction);

private:
    //! Connects between sourceUnit and intermediateUnit by assigning copyUnit
    //! between them
    //! \param originID : sourceUnit ID to connect
    //! \param destID : intermediateUnit ID of destination
    //! \param destInputIndex : input index of this connection to destination
    static void m_connectSourceToHidden(size_t originID, size_t destID,
                                        size_t destInputIndex = 0);

    //! Connects between intermediateUnit and intermediateUnit by assigning
    //! copyUnit between them
    //! \param originID : unique ID of origin intermediateUnit
    //! \param destID : unique ID of destination intermediateUnit
    //! \param destInputIndex : input index of this connection to destination
    static void m_connectHiddenToHidden(size_t originID,
                                        size_t destID,
                                        size_t destInputIndex = 0);

    //! Connects between intermediateUnit and sinkUnit by assigning
    //! copyUnit between them
    //! \param originID : unique ID of origin intermediateUnit
    //! \param destID : unique ID of destination sinkUnit
    //! \param destInputIndex : input index of this connection to destination
    static void m_connectHiddenToSink(size_t originID, size_t destID,
                                      size_t destInputIndex = 0);

    //! Connects unit with previous units using ID
    static void m_connectWithPreviousUnit(
        const std::vector<UnitIdentifier>& previousUnitVector,
        UnitIdentifier subjectUnitIdentifier);

    //! Routine for thread which executes mainUnits
    static void m_run();

    static void m_executeComputeUnits();

    static void m_executeCopyUnits();

    static bool m_IsComplete(size_t epochs);

    static std::thread m_scanMainThread;

    static std::thread m_scanCopyThread;
    /// Stores active threads assigned for executing mainTasks
    static std::vector<std::thread> m_mainThreadPool;
    /// Stores active threads assigned for executing copyTasks
    static std::vector<std::thread> m_copyThreadPool;
    /// TaskQueue storing tasks from SourceUnit, IntermediateUnit and SinkUnit
    static SpinLockQueue<TaskWrapper> m_taskQueue;
    /// True if this engine is active false otherwise
    static bool m_active;

    static std::vector<SharedPtr<SourceUnit>> m_sourceUnitVector;
    static std::vector<SharedPtr<SinkUnit>> m_sinkUnitVector;
    static std::vector<SharedPtr<HiddenUnit>> m_hiddenUnitVector;
    static std::vector<SharedPtr<CopyUnit>> m_copyUnitVector;

    /// number of epochs to run the graph
    /// If stateNum reaches this, that unit will be no longer computed
    static size_t m_maxEpochs;
    static std::atomic_bool m_ready;
};
} // namespace CubbyDNN

#endif  // CAPTAIN_THREADMANAGER_HPP

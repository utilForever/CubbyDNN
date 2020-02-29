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

#include <functional>
#include <thread>
#include <vector>

namespace CubbyDNN
{
//! Task wrapper for sending tasks to pending threads in the thread pool
//! tasks are sent as function pointers and its type
struct TaskWrapper
{
    TaskWrapper()
        : Type(TaskType::None)
    {
    }

    //! Constructor
    //! \param type : type of this task
    //! \param compute : function to execute the computation
    //! \param updateState : function m_objectPtr that updates the unit states
    //! next unit
    //! TODO : Remove updateState if it can be unified to one function
    TaskWrapper(TaskType type, std::function<void()> compute,
                std::function<void()> updateState)
        : Type(type),
          m_compute(std::move(compute)),
          m_updateState(std::move(updateState))
    {
    }

    //! Automatically builds function that will execute main operation and
    //! update its state
    //! \return : lambda including main function and state updater
    std::function<void()> GetTask()
    {
        auto& mainFunc = m_compute;
        auto& updateState = m_updateState;
        const auto combinedFunc = [mainFunc, updateState]()
        {
            mainFunc();
            updateState();
        };
        return combinedFunc;
    }

    //! Gets lambda executing main function
    //! \return : lambda including main function
    std::function<void()> GetPureTask()
    {
        auto& mainFunc = m_compute;
        return [mainFunc]() { mainFunc(); };
    }

    TaskType Type;

private:
    /// Function to execute in the thread queue
    std::function<void(void)> m_compute;
    /// Function used to update state of the ComputableUnit
    std::function<void(void)> m_updateState;
};

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
        return m_mainTaskQueue.Size();
    }

    //! Scans sourceUnitVector, sinkUnitVector, intermediateUnitUnitVector
    //! and pushes units ready to be executed in the mainTaskQueue
    static void ScanUnitTasks();


public:

    //! Execute the graph using single thread
    static void StartExecution(size_t epochs);

    //! Initializes thread pool
    //! \param mainThreadSize : number of threads to assign to main operation
    //! \param copyThreadSize : number of threads to assign to copy operation
    //! \param epochs : number of epochs to execute the graph
    //! concurrency)
    static void StartExecution(size_t mainThreadSize, size_t copyThreadSize,
                               size_t epochs);

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
        UnitIdentifier& unitA, const UnitIdentifier& unitB);


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
        const std::function<void(const Tensor& tensor)>& testFunction);

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
    static void m_runMain();
    //! Routine for thread which executes copy operation
    static void m_runCopy();

    static std::thread m_scanMainThread;

    static std::thread m_scanCopyThread;
    /// Stores active threads assigned for executing mainTasks
    static std::vector<std::thread> m_mainThreadPool;
    /// Stores active threads assigned for executing copyTasks
    static std::vector<std::thread> m_copyThreadPool;
    /// TaskQueue storing tasks from SourceUnit, IntermediateUnit and SinkUnit
    static SpinLockQueue<TaskWrapper> m_mainTaskQueue;
    /// TaskQueue storing tasks from copyUnit
    static SpinLockQueue<TaskWrapper> m_copyTaskQueue;
    /// True if there are any possible nodes that hasn't been queued into
    /// taskQueue False if not
    static std::atomic<bool> m_dirty;
    /// True if this engine is active false otherwise
    static bool m_active;

    static std::vector<SharedPtr<SourceUnit>> m_sourceUnitVector;
    static std::vector<SharedPtr<SinkUnit>> m_sinkUnitVector;
    static std::vector<SharedPtr<HiddenUnit>> m_hiddenUnitVector;
    static std::vector<SharedPtr<CopyUnit>> m_copyUnitVector;

    /// number of epochs to run the graph
    /// If stateNum reaches this, that unit will be no longer computed
    static size_t m_maxEpochs;
};
} // namespace CubbyDNN

#endif  // CAPTAIN_THREADMANAGER_HPP

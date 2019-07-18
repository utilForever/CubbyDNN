// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBYDNN_THREADMANAGER_HPP
#define CUBBYDNN_THREADMANAGER_HPP

#include <cubbydnn/Engine/SpinLockQueue.hpp>
#include <cubbydnn/Utils/SharedPtr-impl.hpp>
#include <cubbydnn/Units/ComputableUnit.hpp>


#include <functional>
#include <thread>
#include <vector>
#include <condition_variable>

namespace CubbyDNN
{
enum class TaskType
{
    ComputeSource,
    ComputeSink,
    ComputeIntermediate,
    Copy,
    Join,
    Empty
};

/**
 * Task wrapper for sending tasks to pending threads in the thread pool
 * tasks are sent as function pointers and its type
 */
struct TaskWrapper
{
    /**
     * Constructor
     * @param type : type of this task
     * @param compute : function to execute the computation
     * @param updateState : function object that updates the unit states
     * @param checkAndInsert : function object that checks previous and next unit
     */
    TaskWrapper(TaskType type, std::function<void(void)> compute,
                std::function<void(void)> updateState)
        : Type(type),
          m_compute(std::move(compute)),
          m_updateState(std::move(updateState))
    {
    }

    std::function<void()> GetTask()
    {
        auto& mainFunc = m_compute;
        auto& updateState = m_updateState;
        auto combinedFunc = [mainFunc, updateState]() {
            mainFunc();
            updateState();
        };
        return combinedFunc;
    }

    TaskType Type;

 private:
    /// Function to execute in the thread queue
    std::function<void(void)> m_compute;
    /// Function used to update state of the ComputableUnit
    std::function<void(void)> m_updateState;

};

/**
 * Singleton static class for maintaining threads that execute the program
 */
class ThreadManager
{
 protected:
    /**
     * Starts thread which waits for tasks to come in
     */
    static void run();

    /**
     * Initializes thread pool
     * @param threadNum : number of threads to spawn (maxed out to hardware
     * concurrency)
     */
    static void InitializeThreadPool(size_t threadNum);

    /**
     * Enqueues tasks into task queue
     * @param task
     */
    static void EnqueueTask(TaskWrapper&& task);

    /**
     * Dequeue tasks from task queue
     * @return
     */
    static TaskWrapper DequeueTask();

    /**
     * Brings back size of the task queue
     * @return : size of task queue
     */
    static size_t TaskQueueSize()
    {
        return m_taskQueue.Size();
    }

    /**
     * Joins threads in the thread queue and terminates the execution
     */
    static void JoinThreads()
    {
        for (auto& thread : m_threadPool)
        {
            if (thread.joinable())
                thread.join();
        }
    }

    static void Scan();

 private:
    static std::vector<std::thread> m_threadPool;

    static SpinLockQueue<TaskWrapper> m_taskQueue;

    /// True if there are any possible nodes that hasn't been queued into taskQueue
    /// False if not
    static std::atomic<bool> m_dirty;
    static bool m_active;

    static std::vector<SourceUnit> m_sourceUnitVector;
    static std::vector<SinkUnit> m_sinkUnitVector;
    static std::vector<IntermediateUnit> m_intermediateUnitVector;
    static std::vector<CopyUnit> m_copyUnitVector;
};

}  // namespace CubbyDNN

#endif  // CAPTAIN_THREADMANAGER_HPP

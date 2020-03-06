// Copyright (c) 2019 Chris Ohk, Justin Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#ifndef CUBBBYDNN_TASKWRAPPER_HPP
#define CUBBYDNN_TASKWRAPPER_HPP

#include <functional>
#include <cubbydnn/Utils/Declarations.hpp>

namespace CubbyDNN
{
//! Task wrapper for sending tasks to pending threads in the thread pool
//! tasks are sent as function pointers and its type
struct TaskWrapper
{
    TaskWrapper() : Type(TaskType::None)
    {
    }

    TaskWrapper(TaskType type) : Type(type)
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
        const auto combinedFunc = [mainFunc, updateState]() {
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
}

#endif
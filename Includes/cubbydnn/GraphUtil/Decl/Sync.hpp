/**
 * Copyright (c) 2019 Chris Ohk, Justin Kim
 * @file : Sync.hpp
 * @brief : helper functions that link TensorObjects and Operations
 */

#ifndef CUBBYDNN_SYNC_HPP
#define CUBBYDNN_SYNC_HPP

#include <memory>
#include <mutex>
#include <condition_variable>

namespace CubbyDNN
{
    class IExecutable
    {
        virtual void Start() = 0;
        virtual void Finish() = 0;
    };

    /**
     * Mtx and Cond_var for controlling synchronization from Operation to Linker
     */
    struct Sync
    {

    public:
        explicit Sync(int waitFor) : m_resetVal(waitFor) , m_counter(waitFor){}
        /**
         * WaitUntilAllFinish
         * Waits until every operation finishes by checking counter is 0
         */
        void WaitUntilAllFinish(){
            std::unique_lock<std::mutex> lock(m_mtx);
            auto checkCompleted = [this]() {
                return (m_counter == 0) || m_forceFinish;
            };
            m_condVar.wait(lock, checkCompleted);
        }

        /**
         * ResetCounter
         * Resets counter to initial value
         */
        void ResetCounter(){
            std::unique_lock<std::mutex> lock(m_mtx);
            m_counter = m_resetVal;
        }

        /**
         * NotifyFinish
         * Decrements counter by 1 Which means one operation is finished
         */
        void NotifyFinish()
        {
            std::unique_lock<std::mutex> lock(m_mtx);
            if(m_counter > 0)
                m_counter--;
            m_condVar.notify_all();
        }

        /**
         * ForceFinish
         * Forces synchronization process to Finish
        */
        void ForceFinish()
        {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_forceFinish = true;
            m_condVar.notify_all();
        }

    private:

        int m_resetVal;
        int m_counter;
        std::mutex m_mtx;
        std::condition_variable m_condVar;
        bool m_forceFinish = false;
    };

    using SyncPtr = Sync*;
}
#endif //CUBBYDNN_SYNC_HPP

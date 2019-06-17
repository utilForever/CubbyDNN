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
    /**
     * Mtx and Cond_var for controlling synchronization from Operation to Linker
     */
    struct Sync
    {
        explicit Sync(int waitFor) : resetVal(waitFor) , counter(waitFor){}

        int resetVal;
        std::atomic<int> counter;
        std::mutex mtx;
        std::condition_variable condVar;

        /**
         * Waits until every operation finishes by checking counter is 0
         */
        void WaitUntilAllFinish(){
            std::unique_lock<std::mutex> lock(mtx);
            auto checkCompleted = [this]() {
                return counter == 0;
            };
            condVar.wait(lock, checkCompleted);
        }

        /**
         * Resets counter to initial value
         */
        void ResetCounter(){
            counter = resetVal;
        }

        /**
         * Decrements counter by 1 Which means one operation is finished
         */
        void NotifyFinish()
        {
            std::unique_lock<std::mutex> lock(mtx);
            if(counter > 0)
                counter--;
            condVar.notify_all();
        }
    };

    using SyncPtr = Sync*;
}
#endif //CUBBYDNN_SYNC_HPP

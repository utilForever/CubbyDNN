//
// Created by jwkim98 on 6/26/19.
//

#pragma once
#ifndef CAPTAIN_THREADMANAGER_HPP
#define CAPTAIN_THREADMANAGER_HPP

#include <captain/Engine/SpinLockQueue.hpp>

#include <functional>
#include <thread>
#include <vector>

namespace Captain {

enum class TaskType {
  ComputeICF,
  ComputeLC,
  MoveFromPreviousUnit,
  MoveFromWaitQueue,
  GenerateVehicles,
  AcceptVehicles,
  Join,
  Empty
};

/**
 * Task wrapper for sending tasks to pending threads in the thread pool
 * tasks are sent as function pointers and its type
 */
struct Task {
  Task(TaskType type, std::function<void(void)> function)
      : Type(type), Function(std::move(function)) {}

  explicit Task() : Type(TaskType::Empty) {}
  TaskType Type;
  std::function<void(void)> Function;
};

/**
 * Singleton static class for maintaining threads that execute the program
 */
class ThreadManager {
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
  static void EnqueueTask(Task &&task);

  /**
   * Dequeue tasks from task queue
   * @return
   */
  static Task DequeueTask();

  static size_t TaskQueueSize() { return m_task.Size(); }

  static void JoinThreads(){
      for(auto& thread : m_threadPool){
          if(thread.joinable())
            thread.join();
      }
  }

private:
  static std::vector<std::thread> m_threadPool;

  static SpinLockQueue<Task> m_task;
};

} // namespace Captain

#endif // CAPTAIN_THREADMANAGER_HPP

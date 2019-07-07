//
// Created by jwkim98 on 6/26/19.
//
#include <captain/Engine/ThreadManager.hpp>
#include <iostream>

namespace Captain {

std::vector<std::thread> ThreadManager::m_threadPool =
    std::vector<std::thread>();

SpinLockQueue<Task> ThreadManager::m_task = SpinLockQueue<Task>(10000);

void ThreadManager::InitializeThreadPool(size_t threadNum) {
  auto hardwareThreads = std::thread::hardware_concurrency();
  auto activeThreadNum =
      (threadNum < hardwareThreads) ? threadNum : hardwareThreads;
  std::cout << "Creating " << activeThreadNum << " threads" << std::endl;
  m_threadPool.reserve(activeThreadNum);
  for (unsigned count = 0; count < hardwareThreads; ++count) {
    m_threadPool.emplace_back(std::thread(run));
  }
}

// TODO : turn this into spinlock based pending function
void ThreadManager::run() {
  Task task;

  while ((task = m_task.Dequeue()).Type != TaskType::Join) {
    task.Function();
  }
}

void ThreadManager::EnqueueTask(Task &&task) {
  m_task.Enqueue(std::move(task));
}

Task ThreadManager::DequeueTask() { return m_task.Dequeue(); }
} // namespace Captain
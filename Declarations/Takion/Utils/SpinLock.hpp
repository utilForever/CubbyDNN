// Copyright (c) 2020, Jaewoo Kim

// We are making my contributions/submissions to this project solely in our
// personal capacity and are not conveying any rights to any intellectual
// property of any third parties.

#include <Takion/Utils/SpinLock-Decl.hpp>
#include <thread>

namespace Takion {

template <typename T> void SpinLock<T>::ExclusiveLock() {

  while (m_exclusiveAccess.exchange(true, std::memory_order_acquire)) {
  }

  while (m_referenceCount != 0) {
  }
}

template <typename T> void SpinLock<T>::ExclusiveRelease() {
  m_exclusiveAccess.exchange(false, std::memory_order_release);
}

template <typename T> void SpinLock<T>::SharedLock() {
  bool success = false;
  while (!success) {
    while (m_exclusiveAccess == true) {
      std::this_thread::yield();
    }

    m_referenceCount.fetch_add(1, std::memory_order_acquire);

    if (m_exclusiveAccess == true) {
      m_referenceCount.fetch_sub(1, std::memory_order_release);
    } else
      success = true;
  }
}

template <typename T> void SpinLock<T>::SharedRelease() {
  m_referenceCount.fetch_sub(1, std::memory_order_release);
}

} // namespace Captain
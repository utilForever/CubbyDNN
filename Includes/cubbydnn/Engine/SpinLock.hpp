//
// Created by jwkim98 on 6/27/19.
//

#include <captain/Engine/SpinLock-Decl.hpp>

namespace Captain {

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
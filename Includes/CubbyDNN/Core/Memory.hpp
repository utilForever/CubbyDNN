#ifndef CUBBYDNN_MEMORY_HPP
#define CUBBYDNN_MEMORY_HPP

#include <CubbyDNN/Core/Span.hpp>

#include <memory>

namespace CubbyDNN::Core
{
template <typename T>
class Memory
{
 public:
    Memory();
    Memory(std::size_t size);
    Memory(const Memory& rhs);
    Memory(Memory&& rhs) noexcept;
    ~Memory() noexcept = default;

    Memory& operator=(const Memory& rhs);
    Memory& operator=(Memory&& rhs) noexcept;

    std::size_t Size() const noexcept;
    std::size_t Capacity() const noexcept;
    Span<T> GetSpan() const noexcept;
    void Resize(std::size_t size);
    void Reserve(std::size_t capacity);

    template <typename T>
    friend void Swap(Memory<T>& left, Memory<T>& right) noexcept;

 private:
    std::size_t m_size;
    std::size_t m_capacity;
    std::unique_ptr<T[]> m_pointer;
};
}  // namespace CubbyDNN::Core

#include <CubbyDNN/Core/Memory-Impl.hpp>

#endif
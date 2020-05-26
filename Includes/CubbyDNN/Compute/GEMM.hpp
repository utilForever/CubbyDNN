#ifndef CUBBYDNN_GEMM_HPP
#define CUBBYDNN_GEMM_HPP

#include <CubbyDNN/Core/Span.hpp>

namespace CubbyDNN::Compute
{
class GEMM final
{
 public:
    GEMM() = delete;
    ~GEMM() noexcept = delete;
    GEMM(const GEMM& rhs) = delete;
    GEMM(GEMM&& rhs) noexcept = delete;

    GEMM& operator=(const GEMM& rhs) = delete;
    GEMM& operator=(GEMM&& rhs) noexcept = delete;

    static void __vectorcall Multiply(std::size_t maxIndex, std::size_t numRow,
                                      std::size_t numColumn,
                                      const Core::Span<float> left,
                                      const Core::Span<float> right,
                                      Core::Span<float> destination) noexcept;

    static void __vectorcall MultiplyAdd(
        std::size_t maxIndex, std::size_t numRow, std::size_t numColumn,
        const Core::Span<float> left, const Core::Span<float> right,
        Core::Span<float> destination) noexcept;

    static void __vectorcall dMultiplyLeft(
        std::size_t maxIndex, std::size_t numRow, std::size_t numColumn,
        const Core::Span<float> gradient, const Core::Span<float> right,
        Core::Span<float> destination) noexcept;

    static void __vectorcall dMultiplyAddLeft(
        std::size_t maxIndex, std::size_t numRow, std::size_t numColumn,
        const Core::Span<float> gradient, const Core::Span<float> right,
        Core::Span<float> destination) noexcept;

    static void __vectorcall dMultiplyAddRight(
        std::size_t maxIndex, std::size_t numRow, std::size_t numColumn,
        const Core::Span<float> gradient, const Core::Span<float> left,
        Core::Span<float> destination) noexcept;
};
}  // namespace CubbyDNN::Compute

#endif
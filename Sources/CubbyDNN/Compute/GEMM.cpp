#include <CubbyDNN/Compute/GEMM.hpp>

#include <vector>

#include <immintrin.h>

namespace CubbyDNN::Compute
{
void __vectorcall GEMM::Multiply(std::size_t maxIndex, std::size_t numRow,
                                 std::size_t numColumn,
                                 const Core::Span<float> left,
                                 const Core::Span<float> right,
                                 Core::Span<float> destination) noexcept
{
    destination.FillZero();

    MultiplyAdd(maxIndex, numRow, numColumn, left, right, destination);
}

void __vectorcall GEMM::MultiplyAdd(std::size_t maxIndex, std::size_t numRow,
                                    std::size_t numColumn,
                                    const Core::Span<float> left,
                                    const Core::Span<float> right,
                                    Core::Span<float> destination) noexcept
{
    const auto* l = left.begin();
    const auto* r = right.begin();
    auto* __restrict d = destination.begin();

#pragma omp parallel for schedule(guided) default(shared)                    \
    num_threads(static_cast <int>(std::max <std::size_t>(                    \
        1u, std::min <std::size_t>(maxIndex * numRow * numColumn / 1600000u, \
                                   std::thread::hardware_concurrency()))))

    for (std::int64_t numR = 0; numR < static_cast<std::int64_t>(numRow);
         ++numR)
    {
        for (std::size_t numC = 0; numC < numColumn; ++numC)
        {
            std::size_t numIndex = 0;
            auto sum = _mm256_setzero_ps();

            for (; numIndex + 8 <= maxIndex; numIndex += 8)
            {
                sum = _mm256_fmadd_ps(
                    _mm256_loadu_ps(l + numR * maxIndex + numIndex),
                    _mm256_loadu_ps(r + numC * maxIndex + numIndex), sum);
            }

            const auto sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1),
                                           _mm256_castps256_ps128(sum));
            const auto sum64 =
                _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            const auto sum32 =
                _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
            auto numSum = _mm_cvtss_f32(sum32);

            for (; numIndex < maxIndex; ++numIndex)
            {
                numSum += l[numR * maxIndex + numIndex] *
                          r[numC * maxIndex + numIndex];
            }

            d[numR * numColumn + numC] += numSum;
        }
    }
}

void __vectorcall GEMM::dMultiplyAddLeft(std::size_t maxIndex,
                                         std::size_t numRow,
                                         std::size_t numColumn,
                                         const Core::Span<float> gradient,
                                         const Core::Span<float> right,
                                         Core::Span<float> destination) noexcept
{
    const auto* g = gradient.begin();
    const auto* r = right.begin();
    auto* __restrict d = destination.begin();

    std::vector<float> rightTransposed(maxIndex * numColumn);
    auto* __restrict rTransposed = rightTransposed.data();

#pragma omp parallel default(shared)                                \
    num_threads(static_cast <int>(std::max <std::size_t>(           \
        1u, std::min <std::size_t>(maxIndex * numColumn / 1600000u, \
                                   std::thread::hardware_concurrency()))))
    {
#pragma omp for schedule(guided)
        for (std::int64_t numR = 0; numR < static_cast<std::int64_t>(numColumn);
             ++numR)
        {
            for (std::size_t numC = 0; numC < maxIndex; ++numC)
            {
                rTransposed[numC * numColumn + numR] =
                    r[numR * maxIndex + numC];
            }
        }

#pragma omp for schedule(guided)
        for (std::int64_t numR = 0; numR < static_cast<std::int64_t>(numRow);
             ++numR)
        {
            for (std::size_t numC = 0; numC < maxIndex; ++numC)
            {
                std::size_t numIndex = 0;
                auto sum = _mm256_setzero_ps();

                for (; numIndex + 8 <= numColumn; numIndex += 8)
                {
                    sum = _mm256_fmadd_ps(
                        _mm256_loadu_ps(g + numR * numColumn + numIndex),
                        _mm256_loadu_ps(rTransposed + numC * numColumn +
                                        numIndex),
                        sum);
                }

                const auto sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 1),
                                               _mm256_castps256_ps128(sum));
                const auto sum64 =
                    _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                const auto sum32 =
                    _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
                auto numSum = _mm_cvtss_f32(sum32);

                for (; numIndex < numColumn; ++numIndex)
                {
                    numSum += g[numR * numColumn + numIndex] *
                              rTransposed[numC * numColumn + numIndex];
                }

                d[numR * maxIndex + numC] += numSum;
            }
        }
    }
}
}  // namespace CubbyDNN::Compute
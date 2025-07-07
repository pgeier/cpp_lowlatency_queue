#pragma once

#if (defined __cplusplus) && (__cplusplus >= 201703L)

#include <limits.h>  // import CHAR_BIT
#include <limits>
#include <type_traits>

namespace llq {

//==============================================================================

// Next power of two
// End condition
template <typename IntType, std::size_t d, std::enable_if_t<(d >= (sizeof(IntType)) * CHAR_BIT), bool> = true>
constexpr inline IntType nextPowOf2(IntType v, std::integral_constant<std::size_t, d>) noexcept {
    return ++v;
}

// Next power of two: compile-time recursion
template <typename IntType, std::size_t d, std::enable_if_t<(d < (sizeof(IntType)) * CHAR_BIT), bool> = true>
constexpr inline IntType nextPowOf2(IntType v, std::integral_constant<std::size_t, d>) noexcept {
    return nextPowOf2(v | (v >> d), std::integral_constant<std::size_t, (d << 1)>{});
}

// Next power of two
// Start condition
template <typename IntType>
constexpr inline IntType nextPowOf2(IntType v) noexcept {
    static_assert(std::is_integral_v<IntType> && !std::numeric_limits<IntType>::is_signed,
                  "nextPowOf2 is only valid for unsigned integral types");
    return nextPowOf2(--v, std::integral_constant<std::size_t, 1>{});
};

//==============================================================================

// prev power of two
// Start condition
template <typename IntType>
constexpr inline IntType prevPowOf2(IntType v) noexcept {
    static_assert(std::is_integral_v<IntType> && !std::numeric_limits<IntType>::is_signed,
                  "prevPowOf2 is only valid for unsigned integral types");
    const IntType np2 = nextPowOf2(v);
    return (v == np2) ? v : (np2 >> 1);
};

//==============================================================================

// Compute bit width by using nextPowOf2
// End condition
template <typename IntType, std::size_t w, std::enable_if_t<(w == 1), bool> = true>
constexpr inline IntType bitWidth(IntType v, IntType s, std::integral_constant<std::size_t, w>) noexcept {
    return s;
}

// Compute bit width by using nextPowOf2
// compile-time recursion - the value v is expected to have only one bit set
// (power of two...). Hence splitting half of the value recursively allows
// determining the number of bits required to represent the number with a fixed
// amount of computations.
template <typename IntType, std::size_t w, std::enable_if_t<(w > 1), bool> = true>
constexpr inline IntType bitWidth(IntType v, IntType s, std::integral_constant<std::size_t, w>) noexcept {
    constexpr std::size_t shift = w >> 1;
    const IntType mask = (1UL << shift) - 1;
    const IntType high = v & (mask << shift);
    const IntType low = v & mask;
    return high ? bitWidth(high >> shift, s + (IntType)shift, std::integral_constant<std::size_t, shift>{})
                : bitWidth(low, s, std::integral_constant<std::size_t, shift>{});
}

// Compute bit width by using nextPowOf2
// Start condition
template <typename IntType>
constexpr inline IntType bitWidth(IntType v) noexcept {
    static_assert(std::is_integral_v<IntType> && !std::numeric_limits<IntType>::is_signed,
                  "bitWidth is only valid for unsigned integral types");
    // Sort out edge case when highest bit is set
    IntType highestNP2 = 1UL << (sizeof(IntType) * CHAR_BIT - 1);
    if (v == highestNP2)
        return (sizeof(IntType) * CHAR_BIT - 1);
    if (v > highestNP2) {
        return sizeof(IntType) * CHAR_BIT;
    }
    const IntType np2 = nextPowOf2(v);
    return bitWidth(v == np2 ? (np2 << 1) : np2, (IntType)0,
                    std::integral_constant<std::size_t, sizeof(IntType) * CHAR_BIT>{});
};

//==============================================================================

}  // namespace llq

#endif

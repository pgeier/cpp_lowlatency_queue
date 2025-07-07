#pragma once

// ---- LIKELY/UNLIKELY ----
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#elif (defined __has_cpp_attribute) && __has_cpp_attribute(likely)
#define LIKELY(expr) (expr) [[LIKELY]]
#else
#define LIKELY(expr) (expr)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
#elif (defined __has_cpp_attribute) && __has_cpp_attribute(unlikely)
#define UNLIKELY(expr) (expr) [[UNLIKELY]]
#else
#define UNLIKELY(expr) (expr)
#endif

// ---- Cacheline size (C++17) ----
// https://en.cppreference.com/w/cpp/thread/hardware_destructive_interference_size
#include <new>

#ifdef __cpp_lib_hardware_interference_size
#ifndef CACHELINE_SIZE_DESTRUCTIVE
using std::hardware_destructive_interference_size;
#define CACHELINE_SIZE_DESTRUCTIVE std::hardware_destructive_interference_size
#endif
#ifndef CACHELINE_SIZE_CONSTRUCTIVE
using std::hardware_constructive_interference_size;
#define CACHELINE_SIZE_CONSTRUCTIVE std::hardware_constructive_interference_size
#endif
#else
// 64 bytes on x86-64 ....
#ifndef CACHELINE_SIZE_DESTRUCTIVE
constexpr std::size_t hardware_destructive_interference_size = 64;
#define CACHELINE_SIZE_DESTRUCTIVE hardware_destructive_interference_size
#endif
#ifndef CACHELINE_SIZE_CONSTRUCTIVE
constexpr std::size_t hardware_constructive_interference_size = 64;
#define CACHELINE_SIZE_CONSTRUCTIVE hardware_constructive_interference_size
#endif
#endif

#define ALIGN_CACHELINE alignas(hardware_destructive_interference_size)
#define FITS_IN_CACHELINE(ST) (ST <= hardware_constructive_interference_size)
#define ELEMENTS_PER_CACHELINE(ST) (hardware_constructive_interference_size / ST)
#define ELEMENTS_PER_CACHELINE_CEIL(ST) (1 + (hardware_constructive_interference_size - 1) / ST)
#define ELEMENTS_PER_CACHELINE_FLOOR(ST) (hardware_constructive_interference_size - 1) / ST

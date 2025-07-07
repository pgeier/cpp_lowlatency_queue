#pragma once

#if (defined __cplusplus) && (__cplusplus >= 201703L)

#include "AlignedAllocator.h"
#include "BinaryUtils.h"
#include "CompilerAbstractions.h"

#include <array>
#include <atomic>
#include <chrono>
#include <limits>  // INCLUDE CHAR_BIT
#include <optional>
#include <thread>
#include <type_traits>
#include <vector>

#include <condition_variable>
#include <mutex>

#include <exception>

namespace llq {

/**
 * This queue is a circular buffer with atomitc synchronization (no mutex, no
 * kernel). It offers an optimistic push/pop that is very fast due to creating
 * tickets with fetch-and-add (FAA) instructions and avoiding loops over
 * compare-and-swap (CAS) operations.
 *
 * There are also pessimistic approaches tryPush/tryPop that can be used as
 * well. However, using the optimistic approach is encouraged. Using both
 * mechanisms is possible but the resulting behaviour is unknown. Under
 * high-contention the pessimistic push/pop are likely to always fail against
 * optimistic ones.
 *
 * Spinning:
 * Of course there are cases where a consumer has to wait for a producer to push
 * some data or where a producer is waiting to pop some data from the queue. In
 * these cases the threads start spinning on a customizeable spin functor to
 * repeatedly test for an event (e.g. data or a free slot). By default the
 * DefaultPushPopSpinFunctor a busy-waiting spin loop is used such that the
 * thread keeps testing continously (low-latency approach).
 *
 * The SpinFunctor can be customized, e.g. to sleep, yield (allow resheduling)
 * or even wait on a condition variable. Luckily, on linux clocks can be used
 * efficently (via virtual ELF dynamic shared objects (vdso)) without kernel
 * involvation. Hence hybrid approaches that first spin for a specific amount of
 * time before involving the kernel can be used.
 *
 * The customized SpinFunctors can be passed on each push/pop call or are
 * queried from a more general PushPopSpinFunctor. The PushPopSpin functor
 * allows customizing a `elementPushed`/`elementPopped` and retrieving a
 * SpinFunctor via `spinOnPush`/`spinOnPop`.
 *
 * Next to the `DefaultPushPopSpinFunctor` a `ConditionedPushPopSpinFunctor` is
 * provided. The `ConditionedPushPopSpinFunctor` is waiting on condition
 * varibale after a spending specific amount of time with busy waiting.
 *
 * Push/Pop:
 * Push and pops are allowed to throw if the move construction of the contained
 * type is throwing. Otherwise they are not ment to throw to avoid using
 * exceptions at all (may be desired in specific use cases). However, it is
 * possible that a Queue gets aborted - then these methods may return `false` or
 * a `nullopt`.
 *
 * In general a pop (`popWithSpinFunctor`) is performed by providing a functor
 * that is called with the moved value - this avoids requiring types to be
 * default constructible or assignable. On top of this function more convenient
 * functions like `popAsOpt` are defined.
 *
 * Other things that are done:
 *  * Alignment to cachesize and padding after a fixed size of elements such
 * that alignment is guaranteed without waisting too much memory.
 *  * Index mapping to access elements over the whole queue in a intermixed
 * order. This tries to avoid false-sharing on frequent push/pop by different
 * threads if the queue is not on capacity limits.
 */

// For super low-latency demands use a buily-wait spin without rescheduling
template <bool RescheduleThreadOnOversubscription = false>
struct SpinThread {
    // @param overSubscribed   Indicates that the queue is massively
    // oversubscribed by a multiple of the queues capacity
    constexpr void operator()(bool overSubscribed) const noexcept {
        // For sufficient large buffers and enough work, this should perform very
        // well because. If there's enough work, you'll notice the yield in
        // performance, hence best is to find appropriate buffersizes such that
        // oversubscription is not happening... For hyberthreaded threads, this can
        // be hard to find. If a thread has enough work, it's possible that a single
        // thread performs 32 operations before another thread takes over. If there
        // are long periods of no work, the user has to identify these and write a
        // custom spinner to do something more reasonable when this happnes.
        if (RescheduleThreadOnOversubscription && overSubscribed) {
            std::this_thread::yield();
        }
    }
};

// For applications that are expecting to have phases of no work, rescheduling
// should happen. Alternatively a sleep_for or sleep_until can be used or more
// complex wakeups through conditional-variables and mutexs can be used. However
// this would be a hybrid approach to switch between high-throughput and no-work
// phases. By Customizing a PushPopSpinFunctor, it is possible to introduce
// global conditional-varibales.
//
// Note: getting the with chrono is very fast on linux because virtual ELF
// dynamic shared objects (vdso) are used.
template <std::size_t MaxSpinMusBeforeYield = 100, bool RescheduleThreadOnOversubscription = false>
struct SpinTimeThread {
    std::chrono::time_point<std::chrono::steady_clock> startSpin = std::chrono::steady_clock::now();

    // If you wish to yield after some time... (measuring time also has its
    // impact, but is not as expensive as rescheduling the thread) If you know
    // your application well and really know that there are cases when to sleep
    // some threads... you have to define you own spin
    // TODO think about allowing introducing a conditional variable. Then more
    // hooks are required to react on ticket changes...
    void operator()(bool overSubscribed) const noexcept {
        if (RescheduleThreadOnOversubscription && overSubscribed)
            std::this_thread::yield();
        auto spinTime = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::microseconds>(spinTime - startSpin).count()
            > MaxSpinMusBeforeYield) {
            std::this_thread::yield();
        }
    }
};

using DefaultSpinFunctor = SpinThread<>;

// Default Functior with methods that get called when an element has been pushed
// or popped. More over this functor is ment to create spin functors to allow
// sharing additional queue states. For example to wait for a global condition
// variable.
template <typename SpinPushFunctor = DefaultSpinFunctor, typename SpinPopFunctor = DefaultSpinFunctor>
struct DefaultPushPopSpinFunctor {
    constexpr void elementPushed() noexcept {
        // Do nothing by default - customize
    }

    constexpr void elementPopped() noexcept {
        // Do nothing by default - customize
    }

    constexpr SpinPushFunctor spinOnPush() noexcept { return SpinPushFunctor{}; }

    constexpr decltype(auto) spinOnPop() noexcept { return SpinPopFunctor{}; }
};

template <std::size_t MaxSpinMusBeforeWait = 100, std::size_t MaxWaitMus = 100000>
struct ConditionedPushPopSpinFunctor {
    std::mutex mPush_;
    std::condition_variable cvPush_;

    std::mutex mPop_;
    std::condition_variable cvPop_;

    constexpr void elementPushed() noexcept {
        // Do nothing by default - customize
        cvPop_.notify_all();
    }

    constexpr void elementPopped() noexcept { cvPush_.notify_all(); }

    constexpr decltype(auto) spinOnPush() noexcept {
        std::chrono::time_point<std::chrono::steady_clock> startSpin = std::chrono::steady_clock::now();
        return [startSpin, this](bool overSubscribed) {
            auto spinTime = std::chrono::steady_clock::now();
            if (overSubscribed
                || (std::chrono::duration_cast<std::chrono::microseconds>(spinTime - startSpin).count()
                    > MaxSpinMusBeforeWait)) {
                // Lock and wait
                std::unique_lock<std::mutex> lock(this->mPush_);
                this->cvPush_.wait_for(lock, std::chrono::microseconds(MaxWaitMus));
                lock.unlock();
            }
        };
    }

    constexpr decltype(auto) spinOnPop() noexcept {
        std::chrono::time_point<std::chrono::steady_clock> startSpin = std::chrono::steady_clock::now();
        return [startSpin, this](bool overSubscribed) {
            auto spinTime = std::chrono::steady_clock::now();
            if (overSubscribed
                || (std::chrono::duration_cast<std::chrono::microseconds>(spinTime - startSpin).count()
                    > MaxSpinMusBeforeWait)) {
                // Lock and wait
                std::unique_lock<std::mutex> lock(this->mPop_);
                this->cvPop_.wait_for(lock, std::chrono::microseconds(MaxWaitMus));
                lock.unlock();
            }
        };
    }
};

enum class QueueResult : int
{
    Aborted = -1,
    Success = 0,
    Closed = 1,
    SpuriousFailure = 2,
};

namespace details {
/* Computes the number of bits that can be shuffled without exceeding a maximal
 * representable value.
 */
template <typename IntType>
constexpr inline IntType maxShuffleBitWidth(IntType maxNumber) noexcept {
    static_assert(std::is_integral_v<IntType> && !std::numeric_limits<IntType>::is_signed,
                  "maxShuffleBitWidth is only valid for unsigned integral types");
    return (maxNumber + 1) == nextPowOf2(maxNumber) ? bitWidth(maxNumber) : (bitWidth(maxNumber) - 1);
};

/**
 *            MaxBits = 7
 *         /^^^^^^^^^^^\
 *   X X X H H H S L L L
 *         \___/ V \___/
 *           |   |   \ NumBits = 3
 *           |   \ Space = 1
 *           \ NumBits = 3
 *
 * The bits marked as `L L L` get swapped with the bits `H H H`.
 *
 * The user has to make sure that the max index is at least represented by a
 * number of width `NumBits*2+space`. Usuallaly the space is zero or might be
 * computed from maxShuffleBitWidth(maxIndex)-2*NumBits
 */
template <typename IntType, std::size_t NumBits>
constexpr inline IntType mapIndex(IntType i, std::size_t space = 0) noexcept {
    static_assert(std::is_integral_v<IntType>, "remapIndex is only valid for integral types");

    constexpr IntType lowerBMask = ((1UL << NumBits) - 1UL);
    const IntType shift = (NumBits + space);

    // Explicit method: masking, shifting and recombining with OR...
    // constexpr IntType lowToHigh = ((i & lowerBMask) << shift);
    // constexpr IntType highToLow = ((i & (lowerBMask << shift)) >> shift);
    // constexpr IntType spaceBetween = (i & (((1 << Space)-1) << b));
    // constexpr IntType higherOrderedBits = (i & ~((1<<MaxBits) -1));
    //
    // return lowToHigh | highToLow | spaceBetween | higherOrderedBits

    // Optimized method using XOR
    // Shift i to the right and perform an XOR, then just mask the lower bits.
    // Result: H H H ^ L L L
    const IntType lowHighMix = (i ^ (i >> shift)) & lowerBMask;

    // Now the mix can be XOR again to restore the original information.
    // For the lower and the higher mixed part, the information will be swapped:
    //  * High part: (H H H ^ L L L) ^ H H H = L L L
    //  * Low part: (H H H ^ L L L) ^ L L L = H H H
    //
    // This works because
    //  * XOR is commutative & associative, e.g.: (H H H ^ L L L) ^ H H H = (H H H
    //  ^ H H H) ^ L L L = 0 0 0 ^ L L L
    //  * 0 is the identity/neutral element: 0 0 0 ^ L L L = L L L
    //
    // For the higher ordered part and the space between, the initial information
    // is preserved as X ^ 0 = X
    return i ^ lowHighMix ^ (lowHighMix << shift);
}

// enum class EntryState : unsigned char {
//     Unused = 0x0,
//     Acquiring = 0x1,
//     Used = 0x2,
//     Releasing = 0x3
// };

template <typename INDT>
inline INDT setTicketUsed(INDT t) noexcept {
    constexpr INDT NUM_BITS = (sizeof(INDT) * CHAR_BIT - 1);
    constexpr INDT HIGH_BIT = (1UL << NUM_BITS);
    return t | HIGH_BIT;
}

template <typename INDT>
inline INDT setTicketUnused(INDT t) noexcept {
    constexpr INDT NUM_BITS = (sizeof(INDT) * CHAR_BIT - 1);
    constexpr INDT HIGH_BIT = (1UL << NUM_BITS);
    constexpr INDT LOW_MASK = HIGH_BIT - 1;
    return (t & LOW_MASK);
}

template <typename INDT>
inline bool isTicketUsed(INDT t) noexcept {
    constexpr std::size_t NUM_BITS = (sizeof(INDT) * CHAR_BIT - 1);
    return (t >> NUM_BITS);
}

// Pack multiple elements in a pack to have more control about cacheline
// boundaries. In some cases the waste of memory can be very high, in this cases
// we accept to spread among more cache line sizes to have a trade of between
// packing and avoiding sharing cache lines. If we put all elemenst in an array
// and just align that array, there is no reason to align at all...
//
//
// Non trivial types need to be stored with a separate state.
// If we put them in a `std::atomic` directly, they would be probably wrapped
// with a mutex - that's what we want to avoid. The alternative of a mutex is a
// state describing the known transitions made with pop/push operations on the
// entry using acquire/release memory order.
//
// Moreover, introducing dedicated states, it is possible to optimistically
// assign entry locations for en/pop operations. This allows oversubscription or
// - in case of empty/full buffers - it is possible to buisy-wait/spin on a
// specific state with low contention instead of spamming head/tail counters
// with a lot of requests.
//
// Note: It is also possible to split the state and the data in an array of
// states and an array of data (to improve padding...).
//   However, the whole reason of doing this is to have more awareness about the
//   memory layout and cacheline boundaries. Putting independent states together
//   will result in more independent elements accessing the same cacheline,
//   hence it is more reasonable to put related state and data together.
//   The effect of padding is very minimal in this case anyway... the loss due
//   to cacheline alignment is much higher. Experimenting with a combined state
//   array the case of a memory benefit actually never showed up.
template <typename T, std::size_t N, bool Align = true, typename INDT = unsigned int>
struct EntryPack {
    // static_assert(std::atomic<EntryState>::is_always_lock_free, "EntryState
    // must be always lock free");
    static_assert(std::atomic<INDT>::is_always_lock_free, "INDT must be always lock free");
    struct None {};

    template <typename U>
    struct DL;

    // Define trivial destructor in specialized base class - allows having
    // constexpr optional for trivial types
    template <typename U, typename Enable = void>
    struct DLBase {
        ~DLBase() { static_cast<DL<U>*>(this)->reset(); }

        // Union with members that have non-trivial destructors need to define a
        // destructor
        union opt_value_type {
            None none;
            U value;
            ~opt_value_type() {}
        };
    };

    template <typename U>
    struct DLBase<U, typename std::enable_if_t<std::is_trivially_destructible<U>::value>> {
        ~DLBase() = default;

        union opt_value_type {
            None none;
            U value;
            ~opt_value_type() = default;
        };
    };

    template <typename U>
    struct DL : DLBase<U> {
        using opt_value_type = typename DLBase<U>::opt_value_type;

        // std::atomic<EntryState> state{EntryState::Unused};
        std::atomic<INDT> ticket;
        // Will be manually initialized by container. Holds the head/tail index
        // that's supposed be read next For scenarios with small buffers & many
        // threads, it is possible that multiple threads compete about the same
        // entry when buffer size has been exceeded.
        // Instead of letting them compete, we prioritize the thread that is waiting
        // longer... Eventually this type of contention should not happen to
        // often... However expiremnts on a usual quadcore (Mac M1) showed weird
        // behaviour (test with 4 producer & 4 consumer constantly enqueuing and
        // dequeing N strings):
        //   It is very likely that a consumer optimistically increments the tail
        //   and then gets suspended for a long time. Inbetween other consumers and
        //   threads are rolling over the buffer multiple times. As a result,
        //   sometimes the last thread in action got stuck because he hit an empty
        //   entry and waited for a producer to push. I guess the thread got
        //   supsended after fetching an tail index. Meanwhile the tail gets
        //   incremented... once the thread is upagain and performed the last deqeue
        //   operation. The next pop by chance just wants to access the same index
        //   again... I could not figure out how the it lead directly to this
        //   constellation... but it happened even more often for smaller buffers.
        //
        //   For long production runs where threads spin on queues until they get
        //   aborted or closed externally... that doesn't matter. However In a
        //   nitty-gritty test where we expect to perform exactly N push & pop, this
        //   is an important constraint.
        opt_value_type value{None{}};

        void reset() noexcept {
            // Non atomic, resetting only performed on destruction
            // A clear state (unused or used, but not Acquiring or Releasing) needs to
            // be assumed
            const auto t = ticket.load(std::memory_order_relaxed);
            if (isTicketUsed(t)) {
                // state.store(EntryState::Unused, std::memory_order_relaxed);
                ticket.store(setTicketUnused(t), std::memory_order_relaxed);
                value.value.~U();
                value.none = None{};
            }
        }
    };

    using EntryType = DL<T>;
    using DataType = std::array<EntryType, N>;

    // Value initialization
    alignas(Align ? hardware_destructive_interference_size : std::alignment_of_v<DataType>) DataType data = {};

    // static constexpr bool isComposed = true;

    template <typename Visitor>
    decltype(auto) visit(std::size_t ind, Visitor&& v) noexcept(noexcept(std::forward<Visitor>(v)(data[ind]))) {
        return std::forward<Visitor>(v)(data[ind]);
    }

    EntryPack() noexcept = default;

    EntryPack(const EntryPack&) = delete;

    EntryPack(EntryPack&& other) noexcept(moveAssign(std::declval<EntryPack&&>())) { moveAssign(std::move(other)); };

    EntryPack& operator=(const EntryPack&) = delete;

    EntryPack& operator=(EntryPack&& other) noexcept(noexcept(moveAssign(std::declval<EntryPack&&>()))) {
        moveAssign(std::move(other));
        return *this;
    }

    //  Non atomic - may not be used when this or other are accessed by multiple
    //  threads... but that should be clear
    void moveAssign(EntryPack&& other) noexcept(std::is_nothrow_move_constructible<T>::value) {
        for (std::size_t i = 0; i < N; ++i) {
            data[i].ticket.exchange(other.data[i].ticket);

            // Can only safely move Used entry states. This does not work if the other
            // is actively used ofc...
            if (isTicketUsed(data[i].tickte)) {
                new (&data[i].value.value) T(std::move(other.data[i].value));
            }
            else {
                data[i].value.none = None{};
            }

            // Reset other... will only be applied when
            other.data[i].reset();
        }
    }
};

template <typename T, typename INDT = unsigned int>
struct EntryTraits {
    static constexpr std::size_t div_ceil(std::size_t a, std::size_t b) { return 1 + (a - 1) / b; }
    // Return the number of bytes per element for a container of elements.
    // If the sie of the EntryPack is larger than `maxMultipleOfCacheline` *
    // CACHELINE_SIZE_CONSTRUCTIVE return numeric_limits::max()
    template <std::size_t numElemPack, std::size_t maxMultipleOfCacheline = 4>
    static constexpr std::size_t densityForAlignedPack() {
        return (sizeof(EntryPack<T, numElemPack, true, INDT>) <= (maxMultipleOfCacheline * CACHELINE_SIZE_CONSTRUCTIVE))
                 ? div_ceil(sizeof(EntryPack<T, numElemPack, true, INDT>), numElemPack)
                 : std::numeric_limits<std::size_t>::max();
    }

    template <typename Arg1>
    static constexpr std::size_t argminWorker(std::size_t minInd, std::size_t runInd, Arg1 arg1) {
        return minInd;
    }

    template <typename Arg1, typename Arg2, typename... Args>
    static constexpr std::size_t argminWorker(std::size_t minInd, std::size_t runInd, Arg1 arg1, Arg2 arg2,
                                              Args&&... args) {
        return (arg1 <= arg2) ? argminWorker(minInd, runInd + 1, arg1, args...)
                              : argminWorker(runInd, runInd + 1, arg2, args...);
    }

    template <typename... Args>
    static constexpr std::size_t argmin(Args&&... args) {
        return argminWorker((std::size_t)0, 1, args...);
    }

    // Check how many single unaligned elements would fit into a cacheline and get
    // the min number of bits required to represent the largest index. Example,
    // assume M1 Cacheline of 64Bytes (gcc12):
    //   * Entry size: 80Bytes => Larger than cache line, bit width is 0
    //   * Entry size: 40Bytes => 1 element per cache line. Index to access the
    //   last element is 0. Hence compute bit width of 0, which is 0
    //   * Entry size: 32Bytes => 2 elements per cache line. Index to access the
    //   last element is 1. Hence compute bit width of 1, which is 1
    //   * Entry size: 20Bytes => 3 elements per cache line. Index to access the
    //   last element is 2. Hence compute bit width of 1, which is 2
    //
    // static constexpr std::size_t MinNumShuffleBits =
    // FITS_IN_CACHELINE(sizeof(EntryPack<T,1,false, INDT>)) ?
    // bitWidth(ELEMENTS_PER_CACHELINE(sizeof(EntryPack<T,1,false, INDT>)) - 1) :
    // 0; static constexpr std::size_t MinContainerSize = 1UL <<
    // (MinNumShuffleBits * 2);

    // Number of shuffle bits when testing different CL sizes
    static constexpr std::size_t TestMinNumShuffleBits
        = FITS_IN_CACHELINE(sizeof(EntryPack<T, 1, false, INDT>))
            ? bitWidth(prevPowOf2(ELEMENTS_PER_CACHELINE(sizeof(EntryPack<T, 1, false, INDT>))) - 1)
            : 0;

    // Test compression for different numbers of packed elements aligned to
    // cacheline.
    //
    // Trivial way: Allocate an array of states and align the whole array to the
    // cacheline - i.e. do not introduce padding between elements.
    //              This has the advantage of saving memory, but elements are
    //              stored across multiple cachelines - i.e. there is no real
    //              benefit from aligning the array at all if the elemnst are not
    //              aligned to the cacheline as well.
    //
    // Aligning each element to the cachline might require too much memory. For
    // small elements it is reasonable to pack multipe elements to fit a whole
    // cacheline. However the compression might be higher if multiple elemenst are
    // also packed to multiple (but not too many) cachelines. But then the
    // justification of performing packing & alignment over just following the
    // trivial approach is blurring. By having an aligned packing, at least cache
    // utilization may be more predictive and we avoid having many overlapping
    // objects.
    //
    // Example, assume M1 Cacheline of 64Bytes (gcc12). Compare entries with
    // (In this examle entries are with unsigned int as ticket/state type)
    //   * Type int:
    //      * Size of EntryPack<_,1,false,INDT>: 8Bytes                  -> 8
    //      element per cache line, TestMinNumShuffleBits = bitWidth(pp2(8)-1) =
    //      bitWidth(7) = 3
    //      * Size of EntryPack<_,8,true,INDT>: 64Bytes    (1x CL)       ->
    //      density: 8BytePerElem
    //   * Type tuple<int,int>:
    //      * Size of EntryPack<_,1,false,INDT>: 12Bytes                 -> 5
    //      element per cache line, TestMinNumShuffleBits = bitWidth(pp2(5)-1) =
    //      bitWidth(3) = 2
    //      * Size of EntryPack<_,4,true,INDT>: 64Bytes    (1x CL)       ->
    //      density: 16BytePerElem
    //      * Size of EntryPack<_,8,true,INDT>: 128Bytes   (2x CL)       ->
    //      density: 16BytePerElem
    //      * Size of EntryPack<_,16,true,INDT>: 192Bytes  (3x CL)       ->
    //      density: 12BytePerElem (Least common multiple between 3xCacheline and
    //      type size)
    //   * Type tuple<int,int,int>:
    //      * Size of EntryPack<_,1,false,INDT>: 16Bytes                 -> 4
    //      element per cache line, TestMinNumShuffleBits = bitWidth(pp2(4)-1) =
    //      bitWidth(3) = 2
    //      * Size of EntryPack<_,4,true,INDT>: 64Bytes    (1x CL)       ->
    //      density: 16BytePerElem -> Take this packing
    //      * Size of EntryPack<_,8,true,INDT>: 128Bytes   (2x CL)       ->
    //      density: 16BytePerElem
    //   * Type tuple<int,int,int,int>:
    //      * Size of EntryPack<_,1,false,INDT>: 20Bytes                 -> 3
    //      element per cache line, TestMinNumShuffleBits = bitWidth(pp2(3)-1) =
    //      bitWidth(1) = 1
    //      * Size of EntryPack<_,1,true,INDT>: 64Bytes    (1x CL)       ->
    //      density: 64BytePerElem
    //      * Size of EntryPack<_,2,true,INDT>: 64Bytes    (1x CL)       ->
    //      density: 32BytePerElem
    //      * Size of EntryPack<_,4,true,INDT>: 128Bytes   (2x CL)       ->
    //      density: 32BytePerElem
    //      * Size of EntryPack<_,8,true,INDT>: 192Bytes   (3x CL)       ->
    //      density: 24BytePerElem
    //      * Size of EntryPack<_,16,true,INDT>: 320Bytes  (5x CL)       ->
    //      density: 20BytePerElem (Least common multiple between 5xCacheline and
    //      type size)
    //   * Type tuple<int,int,int,int,int>:
    //      * Size of EntryPack<_,1,false,INDT>: 24Bytes                 -> 2
    //      element per cache line, TestMinNumShuffleBits = bitWidth(pp2(2)-1) =
    //      bitWidth(1) = 1
    //      * Size of EntryPack<_,2,true,INDT>: 64Bytes    (1x CL)       ->
    //      density: 32BytePerElem
    //      * Size of EntryPack<_,4,true,INDT>: 128Bytes   (2x CL)       ->
    //      density: 32BytePerElem
    //      * Size of EntryPack<_,8,true,INDT>: 192Bytes   (3x CL)       ->
    //      density: 24BytePerElem (Least common multiple between 3xCacheline and
    //      type size)
    //   * Type tuple<int,int,int,int,int,int>:
    //      * Size of EntryPack<_,1,false,INDT>: 28Bytes                 -> 2
    //      element per cache line, TestMinNumShuffleBits = bitWidth(pp2(2)-1) =
    //      bitWidth(1) = 1
    //      * Size of EntryPack<_,2,true,INDT>: 64Bytes    (1x CL)       ->
    //      density: 32BytePerElem
    //      * Size of EntryPack<_,4,true,INDT>: 128Bytes   (2x CL)       ->
    //      density: 32BytePerElem
    //      * Size of EntryPack<_,8,true,INDT>: 256Bytes   (4x CL)       ->
    //      density: 32BytePerElem
    //      * Size of EntryPack<_,16,true,INDT>: 448Bytes  (7x CL)       ->
    //      density: 28BytePerElem  (Least common multiple between 7xCacheline and
    //      type size)
    //   * Type tuple<int,int,int,int,int,int,int,int>:
    //      * Size of EntryPack<_,1,false,INDT>: 36Bytes                 -> 1
    //      element per cache line, TestMinNumShuffleBits = bitWidth(pp2(1)-1) =
    //      bitWidth(0) = 0
    //      * Size of EntryPack<_,1,true,INDT>: 64Bytes    (1x CL)       ->
    //      density: 64BytePerElem
    //      * Size of EntryPack<_,2,true,INDT>: 128Bytes   (2x CL)       ->
    //      density: 64BytePerElem
    //      * Size of EntryPack<_,4,true,INDT>: 192Bytes   (3x CL)       ->
    //      density: 48BytePerElem
    //      * Size of EntryPack<_,8,true,INDT>: 320Bytes   (5x CL)       ->
    //      density: 40BytePerElem
    //      * Size of EntryPack<_,16,true,INDT>: 576Bytes  (9x CL)       ->
    //      density: 36BytePerElem (Least common multiple between 9xCacheline and
    //      type size)
    //   * Type tuple<long,long,long,long,long>:
    //      * Size of EntryPack<_,1,false,INDT>: 48Bytes                 -> 1
    //      element per cache line, TestMinNumShuffleBits = bitWidth(pp2(1)-1) =
    //      bitWidth(0) = 0
    //      * Size of EntryPack<_,1,true,INDT>: 64Bytes    (1x CL)       ->
    //      density: 64BytePerElem
    //      * Size of EntryPack<_,2,true,INDT>: 128Bytes   (2x CL)       ->
    //      density: 64BytePerElem
    //      * Size of EntryPack<_,4,true,INDT>: 192Bytes   (3x CL)       ->
    //      density: 48BytePerElem
    //      * Size of EntryPack<_,8,true,INDT>: 384Bytes   (6x CL)       ->
    //      density: 48BytePerElem
    //      * Size of EntryPack<_,16,true,INDT>: 768Bytes  (12x CL)      ->
    //      density: 48BytePerElem (Least common multiple between 9xCacheline and
    //      type size)
    //                -> with state packing 704Bytes  (11x CL)      -> density:
    //                44BytePerElem First time some bytes have been saved through
    //                proper padding...
    //                                                                                        but having 16 elements
    //                                                                                        shared a CL is not what we
    //                                                                                        want
    //   * Type tuple<long,long,long,long,long,long>:
    //      * Size of EntryPack<_,1,false,INDT>: 56Bytes                 -> 1
    //      element per cache line, TestMinNumShuffleBits = bitWidth(pp2(1)-1) =
    //      bitWidth(0) = 0
    //      * Size of EntryPack<_,1,true,INDT>: 64Bytes    (1x CL)       ->
    //      density: 64BytePerElem
    //      * Size of EntryPack<_,2,true,INDT>: 128Bytes   (2x CL)       ->
    //      density: 64BytePerElem
    //      * Size of EntryPack<_,4,true,INDT>: 256Bytes   (4x CL)       ->
    //      density: 64BytePerElem
    //      * Size of EntryPack<_,8,true,INDT>: 448Bytes   (7x CL)       ->
    //      density: 56BytePerElem
    //      * Size of EntryPack<_,16,true,INDT>: 896Bytes  (16x CL)      ->
    //      density: 56BytePerElem (Least common multiple between 9xCacheline and
    //      type size)
    //                -> with state packing 832Bytes  (15x CL)      -> density:
    //                52BytePerElem Second time some bytes have been saved through
    //                proper padding...
    //                                                                                        but having 16 elements
    //                                                                                        shared a CL is not what we
    //                                                                                        want
    //
    // I've just tested some sizes... for larger elements the scheme should repeat
    // but become less important. Ofcourse for larger cache lines the behaviour
    // might be different... However the important thing to learn:
    //   * Often the density drop significantly when a subpacking is allowed to
    //   take place over 3 cache lines.
    //     Most of the times this includes just 4 elements, for small elements
    //     (<=24Bytes) it have been even 8 or 16 (Not that there are still a lot
    //     of unrelated elemnts on the first and third cacheline...). Hence I'm
    //     allowing subpacking up to 4 cachelines to improve alignment.
    //   * combining states to save padding bytes is not important and has no
    //   benefit in this use-case
    //   * For very large elements the cacheline invalidation is no big problem
    //   anyway.
    //     The state is positioned at the beginning - the object will probably
    //     then take the rest of the cacheline and even further cachelines. That
    //     separates states on cachelines anyway. Nevertheless I'll align them...
    //     because the importance of these few missing bytes is less important
    //     compared to the size of large objects. As result the tail of big
    //     objects are more likely not to suffer from false sharing with the state
    //     of another object.
    static constexpr std::size_t MaxMultipleOfCacheline = 4;
    static constexpr std::size_t EstimatedNumShuffleBits
        = TestMinNumShuffleBits
        + argmin(densityForAlignedPack<1 << TestMinNumShuffleBits, MaxMultipleOfCacheline>(),
                 densityForAlignedPack<1 << (TestMinNumShuffleBits + 1), MaxMultipleOfCacheline>(),
                 densityForAlignedPack<1 << (TestMinNumShuffleBits + 2), MaxMultipleOfCacheline>());

    // Estimated number of elements in the sub packing
    static constexpr std::size_t EstimatedSubPackingSize = 1UL << EstimatedNumShuffleBits;
    static constexpr std::size_t EstimatedContainerSize = 1UL << (EstimatedNumShuffleBits * 2);

    // Computation for minimal requirements with potentially more padding but
    // lower container size
    static constexpr std::size_t MinNumShuffleBits
        = TestMinNumShuffleBits
        + argmin(densityForAlignedPack<1 << TestMinNumShuffleBits, 1>(),
                 densityForAlignedPack<1 << (TestMinNumShuffleBits + 1), 1>(),
                 densityForAlignedPack<1 << (TestMinNumShuffleBits + 2), 1>());

    static constexpr std::size_t MinSubPackingSize = 1UL << MinNumShuffleBits;
    static constexpr std::size_t MinContainerSize = 1UL << (MinNumShuffleBits * 2);
};

template <typename T, class AllocatorOrSize = AlignedAllocator<T, CACHELINE_SIZE_DESTRUCTIVE>,
          typename INDT = unsigned int>
struct EntryContainer;

// Container for dynamically allocated data when the size is just known at
// runtime
// TODO allow using MinContainer size instead of optimize packing?
// TODO Use something else then std::vector?
template <typename T, class Allocator, typename INDT>
struct EntryContainer {
    static constexpr std::size_t ShuffleBits = EntryTraits<T, INDT>::EstimatedNumShuffleBits;
    static constexpr std::size_t SubSize = EntryTraits<T, INDT>::EstimatedSubPackingSize;
    static constexpr std::size_t Size = EntryTraits<T, INDT>::EstimatedContainerSize;

    using SubPackageType = EntryPack<T, SubSize, true, INDT>;
    using Alloc = typename std::allocator_traits<Allocator>::template rebind_alloc<SubPackageType>;

    ALIGN_CACHELINE std::size_t shuffleSpace_;
    std::vector<SubPackageType, Alloc> entries_;

    template <typename Visitor>
    decltype(auto) visit(std::size_t ind, Visitor&& v) noexcept {
        const std::size_t mappedInd = mapIndex<unsigned int, ShuffleBits>(ind, shuffleSpace_);
        return entries_[mappedInd >> ShuffleBits].visit(mappedInd & ((1UL << ShuffleBits) - 1),
                                                        std::forward<Visitor>(v));
    }

    EntryContainer(std::size_t size) noexcept :
        shuffleSpace_(maxShuffleBitWidth(std::max(size, Size) - 1) - ShuffleBits * 2), entries_(std::max(size, Size)) {}

    EntryContainer(const EntryContainer&) = delete;

    EntryContainer(EntryContainer&& other) noexcept :
        shuffleSpace_{other.shuffleSpace_}, entries_{std::move(other.entries_)} {};

    EntryContainer& operator=(const EntryContainer&) = delete;

    EntryContainer& operator=(EntryContainer&& other) noexcept {
        shuffleSpace_ = other.shuffleSpace_;
        entries_ = std::move(other.entries_);
    };

    constexpr std::size_t size() const noexcept { return entries_.size(); }
};

// Container for compile-time fixed sized containers.
// Should not bee to large as a std::array is used
template <typename T, typename IntType, IntType N, typename INDT>
struct EntryContainer<T, std::integral_constant<IntType, N>, INDT> {

    static constexpr bool UseMinShuffleBits
        = (EntryTraits<T, INDT>::MinContainerSize < EntryTraits<T, INDT>::EstimatedContainerSize)
       && (N < EntryTraits<T, INDT>::EstimatedContainerSize);

    static constexpr std::size_t Size = UseMinShuffleBits ? std::max(N, EntryTraits<T, INDT>::MinContainerSize)
                                                          : std::max(N, EntryTraits<T, INDT>::EstimatedContainerSize);
    static constexpr std::size_t ShuffleBits
        = UseMinShuffleBits ? EntryTraits<T, INDT>::MinNumShuffleBits : EntryTraits<T, INDT>::EstimatedNumShuffleBits;
    static constexpr std::size_t SubSize
        = UseMinShuffleBits ? EntryTraits<T, INDT>::MinSubPackingSize : EntryTraits<T, INDT>::EstimatedSubPackingSize;
    static constexpr std::size_t ShuffleSpace = maxShuffleBitWidth(Size - 1) - ShuffleBits * 2;

    using SubPackageType = EntryPack<T, SubSize, true, INDT>;

    ALIGN_CACHELINE std::array<SubPackageType, Size> entries_;

    template <typename Visitor>
    decltype(auto) visit(std::size_t ind, Visitor&& v) noexcept {
        const std::size_t mappedInd = mapIndex<unsigned int, ShuffleBits>(ind, ShuffleSpace);
        return entries_[mappedInd >> ShuffleBits].visit(mappedInd & ((1UL << ShuffleBits) - 1),
                                                        std::forward<Visitor>(v));
    }

    EntryContainer() noexcept : entries_{} {}

    EntryContainer(const EntryContainer&) = delete;

    EntryContainer(EntryContainer&& other) noexcept {
        for (IntType i = 0; i < N; ++i) {
            entries_[i] = std::move(other.entries_[i]);
        }
    };

    EntryContainer& operator=(const EntryContainer&) = delete;

    EntryContainer& operator=(EntryContainer&& other) noexcept {
        for (IntType i = 0; i < N; ++i) {
            entries_[i] = std::move(other.entries_[i]);
        }
    };

    constexpr std::size_t size() const noexcept { return Size; }
};

/* Multiple-producer-multiple-consumer circular buffer based on atomic
 * synchronization.
 *
 * As system calls and memory allocation is avoided, the buffer is expected to
 * show low latency. The size of the buffer should be larger than the size of
 * producers and consumers. Otherwise oversubscription happens and multpile
 * producers/consumers will content for the same entry. Moreover chosing a power
 * of two as size is always a good option to speed up some arithmetic
 * computations.
 *
 * The buffer is caring about cache alignment and packs multiple entries with a
 * value state to have some kind of trade-off between introducing memory padding
 * for alignment and having a compact memory usage. To avoid subsequent memory
 * lookups an the same cacheline, the memory acceses on the buffer are shuffled
 * to happen in a different order. To do this, the size of the buffer might
 * chose a larger size than desired, but is newer smaller.
 *
 * In general, as long as the size of a single entry is not super big, the
 * minimum size of the buffer might be around 16 entries. Example sizes for 64
 * L1 cache:
 *  * Storing int (4Byte) + state -> Entry size of 8 bytes, 8 elements per cache
 * line. For fast shuffeled index access, subpackaging to powers of 2 is done.
 *    Hence 2^3 = 8 values are packed together on 2 cache lines.
 *    For shuffling the minimum size is then 2^6 = 64 entries, which is also
 * optimal
 *  * However for 2 int (8Byte) + state -> Entry size of 12 bytes, 5 elemnts per
 * cacheline For fast shuffeled index access, subpackaging to powers of 2 is
 * done.
 *      * Hence for a minimal container size 2^2 = 4 values are packed together
 * on 1 cache lines. For shuffling the minimum size is then 2^4 = 16 entries
 *      * For more condensed packing instead of 4 elements on 1 CL (16 Bytes per
 * elem), we can also pack 16 elements on 3 cachelines (12 Bytes per elem) This
 * then results in a minimum container size of 256 elements
 *
 *    For static sized container the layout is already adjusted. However, for
 * dynamically sized containers, currently just the more dense packing is used.
 *
 *
 * Comment on FIFO ordering:
 *   It is hard to say whether operations on the buffer are guaranteed to happen
 * in FIFO order. Defining FIFO order is difficult when multiple threads are
 * involved. A lot of atomic operations can be forced to happened in
 * sequentially-consistent order to avoid having the compiler performing
 * reording between operations on different atomic variables.
 *
 *   However this is only important for the operations on as single thread. For
 * an optimistic push/pop an index is incremented atomically. If for example one
 * thread now pushes two values V1 and V2, it increments an head index and then
 * performs atomic operations on a state.
 *
 *   With relaxed ordering one might assume that the first increment could be
 * reordered after the second - for what ever reason, and hence break FIFO order
 * - unless `std::memory_order_seq_cst` is specified to avoid this kind of
 * reordering. However, the operations on the state have acquire/release
 * semantics and hence should disallow this kind of reordering.
 *
 *   From this point of view, if thread 1 first pushs values V1 and then V2,
 * another thread performing two pops is expected to first see V1 and then V2.
 *
 *
 * Comment on difference to linked-lists:
 *   For 20 years a known problem with lock-free queues implemented through
 * lists is a problem known as ABA problem. In short: 1 thread does a lot of
 * things while for another thread nothing appears to have changed. As a result,
 * already freed elements might reappear in the queue and cause memory
 * problems... Typical solutions are hazard pointers or reference counting to
 * safely deallocate these elements...
 *
 *   For this kind of simulations these problems never occur as memory
 * allocation is completely avoided and elements are simply referred to by
 * indices. Basically this kind of queue is rather a set of memory locations
 * protected by lightweight "spin-locks" that get accessed in a round-robin
 * fashion.
 *
 *   An disadvantage is ofcourse that the size has to be fixed while real queues
 * can hold "arbitrarily" many elements. However, the purpose of queues is
 * usually to share work between threads. Large queues are tried to be avoided
 *   through good work-balancing. In some cases it's also desired to limit the
 * size and block the whole pipeline.
 *
 */
template <typename T, class AllocatorOrSize, typename DefaultPushPopSpinFunctor_, typename INDT>
class LowLatencyQueueBase {
public:
    using ContainerType = EntryContainer<T, AllocatorOrSize, INDT>;

private:
    // Keep separate head_ & tail_ counter that do not get reset frequently but
    // only when a certain limit is reached For Size computation it would be
    // easier to use a single size counter to track thisk. However a size counter
    // would need to be updated with a read-modify-write on each push/pop and
    // enforces more contention between consumers and producers

    static_assert(std::is_integral_v<INDT> && !std::numeric_limits<INDT>::is_signed,
                  "Index type must be an unsigned integral types");
    ALIGN_CACHELINE std::atomic<INDT> head_;
    ALIGN_CACHELINE std::atomic<INDT> tail_;
    ALIGN_CACHELINE std::atomic<bool> abort_;
    std::exception_ptr abortReason_;
    ALIGN_CACHELINE std::atomic<bool> close_;
    static_assert(std::atomic<INDT>::is_always_lock_free,
                  "Integer type for head & tail indices must be always lock free");
    static_assert(std::atomic<bool>::is_always_lock_free, "Bool atomic must be always lock free");

    ContainerType container_;

    DefaultPushPopSpinFunctor_ pushPopSpin_;

    // head_ and tail_ are expected to occupy about roughly the half of the
    // representable value range. This is the limit where we apply modulo
    // operation to reset it freuently to prevent overflows in long runs
    const INDT PERIOD;
    const INDT HALF_PERIOD;

    static constexpr INDT FORCE_RESET_PERIOD_EXCEED = 1 << 10;  // 1024

    // The period for resetting the head & tail must be a multiple of the
    // container size Such that for any head & tail index `i` the following
    // statement is true: i % containerSize == (i % PERIOD) % containerSize
    INDT computeResetPeriod(std::size_t containerSize) noexcept {
        std::size_t s = std::numeric_limits<INDT>::max() >> 1;
        std::size_t m = (s % containerSize);

        // Important - always take a value that is <= half of maximum value -
        // i.e. the most left/highest bit should be always unset
        // by doing so we can use the high bit to differentate ticket states
        return s - m;
        //        return (m == 0) ? s : (s + (containerSize - m));
    }

    template <typename IntType>
    static inline constexpr IntType modLikely(IntType l, IntType r) noexcept {
        if LIKELY (!(r & (r - 1))) {
            return (l & (r - 1));
        }
        else {
            if LIKELY (l >= r) {
                return l % r;
            }
            else {
                return l;
            }
        }
    }

    template <typename IntType>
    static inline constexpr IntType modUnlikely(IntType l, IntType r) noexcept {
        if LIKELY (!(r & (r - 1))) {
            return (l & (r - 1));
        }
        else {
            if UNLIKELY (l >= r) {
                return l % r;
            }
            else {
                return l;
            }
        }
    }

    inline INDT modPeriod(INDT l) noexcept {
        if LIKELY (l < PERIOD) {
            return l;
        }
        else {
            return l - PERIOD;
        }
    }

    inline INDT nextUnusedTicket(INDT t, std::size_t containerSize) noexcept {
        return modPeriod(setTicketUnused(t) + containerSize);
    }

public:
    LowLatencyQueueBase(const LowLatencyQueueBase&) = delete;

    LowLatencyQueueBase(LowLatencyQueueBase&& other) noexcept {
        head_.exchange(other.head_);
        tail_.exchange(other.tail_);
        abort_.exchange(other.abort_);
        close_.exchange(other.close_);
        container_ = std::move(other.container_);
        PERIOD = other.PERIOD;
        HALF_PERIOD = other.HALF_PERIOD;
    };

    LowLatencyQueueBase& operator=(const LowLatencyQueueBase& other) = delete;

    LowLatencyQueueBase& operator=(LowLatencyQueueBase&& other) noexcept {
        head_.exchange(other.head_);
        tail_.exchange(other.tail_);
        abort_.exchange(other.abort_);
        close_.exchange(other.close_);
        container_ = std::move(other.container_);
        PERIOD = other.PERIOD;
        HALF_PERIOD = other.HALF_PERIOD;
    };

    template <typename... ContainerArgs>
    LowLatencyQueueBase(ContainerArgs&&... containerArgs) noexcept(noexcept(ContainerType{
        std::forward<ContainerArgs>(containerArgs)...})) :
        head_{0},
        tail_{0},
        abort_{false},
        close_{false},
        container_{std::forward<ContainerArgs>(containerArgs)...},
        PERIOD(computeResetPeriod(container_.size())),
        HALF_PERIOD(PERIOD / 2) {
        // Initialize container tickets
        for (std::size_t i = 0; i < container_.size(); ++i) {
            container_.visit(i, [i, s = container_.size()](auto& entry) {
                entry.ticket.store(i, i + 1 < s ? std::memory_order_relaxed : std::memory_order_release);
            });
        }
    }

    /* Aborts the whole queue - the queue will be not usable anymore.
     *
     * TODO: Think about creating sanitizing feature that resets all tickets and
     * allow reusing the queue
     */
    void abort(std::exception_ptr reason) noexcept {
        if (!abort_.load(std::memory_order_relaxed)) {
            abortReason_ = reason;
            abort_.store(true, std::memory_order_release);
        }
    }
    void abort() noexcept {
        abort(std::make_exception_ptr(std::runtime_error("Queue has been aborted without an exception.")));
    }

    bool hasBeenAborted() const noexcept { return abort_.load(std::memory_order_relaxed); }

    void throwIfAborted() const {
        if (abort_.load(std::memory_order_acquire)) {
            if (abortReason_) {
                std::rethrow_exception(abortReason_);
            }
            else {
                throw std::runtime_error("Queue has been aborted without an exception.");
            }
        }
    }

    /* Closes the whole queue - allows producers to finish and allows consumers to
     * pop until the queue is empty.
     *
     * TODO: Think about creating sanitizing feature that resets all tickets and
     * allow reusing the queue
     */
    void close() noexcept { close_.store(true); }

    bool isClosed() const noexcept { return close_.load(std::memory_order_relaxed); }

private:
    template <typename SpinFunctor, typename... ElemOrArgs>
    QueueResult pushIndex(INDT ticket, SpinFunctor&& spin,
                          ElemOrArgs&&... elemOrArgs) noexcept(noexcept(T(std::forward<ElemOrArgs>(elemOrArgs)...))
                                                               && noexcept(spin(false))) {
        const INDT ind = modLikely(ticket, (INDT)container_.size());
        return container_.visit(ind, [&](auto& entry) {
            // Wait for ticket...
            // one periodic oversubscription on container, this preserves order and
            // avoids some quirks that are hard to explain... Perom load with acquire
            // memory order to make sure no other loads (especially the load of the
            // value...) is not issue before this criteria is met This is to make sure
            // all pop operations have released their stuff are not writing anymore
            INDT currentTicket;
            while
                UNLIKELY((currentTicket = entry.ticket.load(std::memory_order_acquire)) != ticket) {
                    if (abort_.load(std::memory_order_relaxed)) {
                        return QueueResult::Aborted;
                    }
                    // The current ticket is not our ticket - so it must be a previous
                    // ticket. If its not used, we know that a previous push is still
                    // pending...
                    const bool overSubscribed = !isTicketUsed(currentTicket);

                    // Put some flag inside to customize spin, i.e. yield in this case
                    spin(overSubscribed);
                };

            // Now we are save to set the value... no need to CAS
            new (&entry.value.value) T(std::forward<ElemOrArgs>(elemOrArgs)...);

            // Finally set the next ticket
            entry.ticket.store(setTicketUsed(ticket), std::memory_order_release);
            return QueueResult::Success;
        });
    }

    template <typename MoveElemFunctor, typename SpinFunctor>
    QueueResult popIndex(INDT ticket, MoveElemFunctor&& moveElem, SpinFunctor&& spin) noexcept(
        noexcept(std::forward<MoveElemFunctor>(moveElem)(std::move(std::declval<T>()))) && noexcept(spin(false))) {
        const INDT ind = modLikely(ticket, (INDT)container_.size());
        return container_.visit(ind, [&](auto& entry) {
            // Wait for ticket...
            const INDT usedTicket = setTicketUsed(ticket);
            INDT currentTicket;
            while
                UNLIKELY((currentTicket = entry.ticket.load(std::memory_order_acquire)) != usedTicket) {
                    if (abort_.load(std::memory_order_relaxed)) {
                        return QueueResult::Aborted;
                    }
                    if (close_.load(std::memory_order_relaxed)
                        && (computeSize(head_.load(std::memory_order_relaxed), ticket) <= 0)) {
                        return QueueResult::Closed;
                    }

                    // The currentTicket is not equal to our ticket, so a previous push or
                    // pop is still going on
                    const bool overSubscribed = (setTicketUnused(currentTicket) != ticket);
                    spin(overSubscribed);
                };

            // Now we are save to read the value
            std::forward<MoveElemFunctor>(moveElem)(std::move(entry.value.value));
            entry.value.value.~T();

            // Now increment ticket for next push/pop
            entry.ticket.store(nextUnusedTicket(ticket, container_.size()), std::memory_order_release);
            return QueueResult::Success;
        });
    }

    // Optimistic approach is not checking sizes but allows spinnign on states
    template <typename UInt, bool RelaxedPeriodicReset = false>
    INDT nextIndOpt(std::atomic<INDT>& av, const UInt size) noexcept {
        // Optimistic fetch_add
        INDT v = av.fetch_add(1, std::memory_order_relaxed);
        // See comment on LowLatencyQueueBase.
        // We could use `memory_order_seq_cst` to avoid that other increments on the
        // same thread are reordered and disturb fifo order - however due to the
        // following acquire/release semantics on the state, this kind of reordering
        // is disallowed (IMO ... I might be wrong).

        // Check bounds to reset counter frequently
        //
        // Method 1: Spin with compare weak
        //   Using this approach we want to guarantee that once the size limit is
        //   reached, the counter is reset to its modulo counterpart, although
        //   multiple producers/consumers compete about it
        //
        // Method 2 (RelaxedPeriodicReset): compare weak/strong without spinning and
        // hope that one of the following up operations will succeed resetting it.
        //   Exceeding the size is no problem as long as it happens before overflows
        //   occur.
        INDT nextExpectedValue = v + 1;
        if UNLIKELY (nextExpectedValue >= PERIOD) {
            INDT newExpectedValue;
            // try-setting modulo value once (optimistic - other future calls will
            // success) or spin if RelaxedPeriodicReset=false. If the value has not
            // been reset for a while
            do {
                // Compute modulo of nextExpectedValue by difference instead of module
                // (division, multiplication, subtraction) nextExpectedValue is expected
                // to be >= PERIOD
                newExpectedValue = nextExpectedValue - PERIOD;
            } while (!av.compare_exchange_weak(nextExpectedValue, newExpectedValue, std::memory_order_relaxed)
                     && (nextExpectedValue >= PERIOD)
                     && (!RelaxedPeriodicReset || (nextExpectedValue >= (PERIOD + FORCE_RESET_PERIOD_EXCEED))));
        }
        // Modulo can take some time, performing a division, multiplicaiton and
        // subtraction. For powers of two it should be fast anyway return
        // modLikely(v, size);
        return modPeriod(v);
    }

    INDT nextHeadOpt() noexcept { return nextIndOpt(head_, container_.size()); }

    INDT nextTailOpt() noexcept { return nextIndOpt(tail_, container_.size()); }

    long computeSize(INDT h, INDT t) noexcept {
        h = modPeriod(h);
        t = modPeriod(t);
        // We assume that the typical difference between head and tail is much
        // smaller than at least a forth of the representable range. On both - head
        // and tail - a modulo operation might be performed. This never happens at
        // the same time. It is possible that head has been reset and holds a low
        // value while tail holds a high value and vice versa. To handle these cases
        // we assume that the difference in application is never as high has MAX/4
        // (=HALF_PERIOD). That means if we check for a high value like this, we
        // assume that the opposite must be true (PERIOD - diff).
        const auto handlePeriodReset = [&](INDT diff, long sign) {
            return (diff > HALF_PERIOD) ? (static_cast<long>(PERIOD - diff) * (sign * (-1)))
                                        : (static_cast<long>(diff) * sign);
        };

        // Tail is allowed to have higher values then head.
        // Either because head has just been reset through modulo operation,
        // or because there are more optimistic pops than pushs
        return (h >= t) ? handlePeriodReset(h - t, 1L) : handlePeriodReset(t - h, -1L);
    }

    // Pessimistic approach is checking sizes and allow failing if size
    // requirements are not met. Only use this if you want to do something else if
    // the push/depush is failing. If you just want to control how to bypass the
    // waiting/spinning, use the optimistic approach with a customized spin
    // functor.
    //
    // By caring about a roughly correct size computation, it is possible to mix
    // optimistic with pessimistic approaches. I.e. consumers can optimistically
    // pop while producers pessimistically enquueue or vice versa. Of course its
    // also possible that consumers and producers mix the usage of pessimistic and
    // optimistic approaches (i.e. some consumers pop pessimistically, some
    // optimistically). However, doing this might be not reasonable under high
    // contention.
    //
    // Note: Although this approach is trying to avoid oversubscription, it is
    // still possible that two consumers or producers compete about the same
    // state. However, this should be marginal.
    std::optional<INDT> nextHeadPes() noexcept {
        INDT h = head_.load(std::memory_order_relaxed);

        // Try to set increment head...
        // If many other producers are enqueuing optimistically, this is likely to
        // fail
        do {
            if (computeSize(h, tail_.load(std::memory_order_relaxed)) >= container_.size) {
                return std::optional<INDT>{};
            }
        } while (!head_.compare_exchange_weak(h, modPeriod(h + 1), std::memory_order_relaxed));

        return std::optional<INDT>{h};
    }

    std::optional<INDT> nextTailPes() noexcept {
        INDT t = tail_.load(std::memory_order_relaxed);

        // Try to set increment tail...
        // If many other consumers are dequeuing optimistically, this is likely to
        // fail
        do {
            auto size = computeSize(head_.load(std::memory_order_relaxed), t);
            if ((size >= container_.size) || (close_.load(std::memory_order_relaxed) && size <= 0)) {
                return std::optional<INDT>{};
            }
        } while (!tail_.compare_exchange_weak(t, modPeriod(t + 1), std::memory_order_relaxed));

        return std::optional<INDT>{t};
    }

public:
    std::size_t capacity() noexcept { return container_.size(); }

    // Can be negative in case of oversubscription (optimistic pops)
    long size() noexcept {
        return computeSize(head_.load(std::memory_order_relaxed), tail_.load(std::memory_order_relaxed));
    }

    // For debugging - remove later
    std::atomic<INDT>& head() noexcept { return head_; }
    std::atomic<INDT>& tail() noexcept { return tail_; }

    // TODO Remove - for debugging purpose only
    template <typename Func>
    void visitContainer(Func&& f) noexcept {
        for (unsigned int i = 0; i < container_.size(); ++i) {
            container_.visit(i, [&](auto& entry) { f(i, entry); });
        }
    }
    void printContainer() noexcept {
        visitContainer([&](auto ind, auto& entry) { const auto t = entry.ticket.load(std::memory_order_relaxed); });
    }

    /**
     * Optimistically add an entry to the buffer by directly acquiring an entry
     * location without checking the size of the buffer. If the buffer is full it
     * can happen that the thread has to spin and wait for the assigned entry to
     * be released.
     *
     * If the number of producing threads is higher than the size of the buffer,
     * entries can be oversubscribed and multiple producers start to compete for
     * one entry.
     *
     * Default spin is a busy-wait without notifying the thread scheduler or
     * involving any other OS/kernel mechanisms. By customizing the SpinFunctor it
     * is possible to create hybrid mechanisms that first busy-wait and later
     * perform a `std::this_thread::yield()` or subscribe to a condition-variable.
     *
     * @param spin         Function that is called everytime the thread has to
     * spin and wait for an state change of an entry.
     * @param elemOrArgs   Forwards all arguments to the constructor of the
     * element type T. (May perform a copy or move construction or just emplaces a
     * value).
     * @return             QueueResult (self-describing). For optimistic
     * approaches  QueueResult::SpuriousFailure is not expected.
     */
    template <typename SpinFunctor, typename... ElemOrArgs>
    QueueResult pushWithSpinFunctor(SpinFunctor&& spin, ElemOrArgs&&... elemOrArgs) noexcept(
        noexcept(T(std::forward<ElemOrArgs>(elemOrArgs)...)) && noexcept(spin(false))) {
        // Optimistic: get next head atomically... then spin on state
        // Note: By using memory_order_seq_cst a a real FIFO order could be
        // achieved. However in cases where more threads than size of buffer are
        // accessing the container, the contention in the state handling can not
        // guarantee FIFO anyway... Moreover it's hard to define "FIFO" for a
        // concurrent programm
        if (close_.load(std::memory_order_relaxed)) {
            return QueueResult::Closed;
        }
        QueueResult ret
            = pushIndex(nextHeadOpt(), std::forward<SpinFunctor>(spin), std::forward<ElemOrArgs>(elemOrArgs)...);
        if (ret == QueueResult::Success) {
            pushPopSpin_.elementPushed();
        }
        return ret;
    }

    /* Calls `pushWithSpinFunctor` with the default spin functor specified through
     * template argument of the LowLatencyQueue. Default is a busy-wait.
     *
     * @param elemOrArgs   Forwards all arguments to the constructor of the
     * element type T. (May perform a copy or move construction or just emplaces a
     * value).
     * @return             QueueResult (self-describing). For optimistic
     * approaches  QueueResult::SpuriousFailure is not expected.
     */
    template <typename... ElemOrArgs>
    QueueResult push(ElemOrArgs&&... elemOrArgs) noexcept(
        noexcept(pushWithSpinFunctor(pushPopSpin_.spinOnPush(), std::forward<ElemOrArgs>(elemOrArgs)...))) {
        return pushWithSpinFunctor(pushPopSpin_.spinOnPush(), std::forward<ElemOrArgs>(elemOrArgs)...);
    }

    /* Calls `pushWithSpinFunctor` with the default spin functor specified through
     * template argument of the LowLatencyQueue. Default is a busy-wait. Throws if
     * the queue has been aborted.
     *
     * @param elemOrArgs   Forwards all arguments to the constructor of the
     * element type T. (May perform a copy or move construction or just emplaces a
     * value).
     * @return             QueueResult (self-describing). For optimistic
     * approaches  QueueResult::SpuriousFailure is not expected.
     */
    template <typename... ElemOrArgs>
    QueueResult pushOrThrow(ElemOrArgs&&... elemOrArgs) {
        QueueResult ret = push(std::forward<ElemOrArgs>(elemOrArgs)...);
        if (ret == QueueResult::Aborted) {
            throwIfAborted();
        }
        return ret;
    }

    /**
     * Optimistically removes an entry from the buffer by directly acquiring an
     * entry location without checking the size of the buffer. If the buffer is
     * empty it can happen that the thread has to spin and wait for the assigned
     * entry to be filled up.
     *
     * If the number of consuming threads is higher than the size of the buffer,
     * entries can be oversubscribed and multiple consumers start to compete for
     * one entry.
     *
     * Default spin is a busy-wait without notifying the thread scheduler or
     * involving any other OS/kernel mechanisms. By customizing the SpinFunctor it
     * is possible to create hybrid mechanisms that first busy-wait and later
     * perform a `std::this_thread::yield()` or subscribe to a condition-variable.
     *
     * @param moveElem  Functor that gets called with an rvalue of the element to
     * be poped. The user can decide what to do.
     * @param spin      Function that is called everytime the thread has to spin
     * and wait for an state change of an entry.
     * @return          Returns the poped value
     * @return          QueueResult (self-describing). For optimistic approaches
     * QueueResult::SpuriousFailure is not expected.
     */
    template <typename MoveElemFunctor, typename SpinFunctor>
    QueueResult popWithSpinFunctor(MoveElemFunctor&& moveElem, SpinFunctor&& spin) noexcept(
        noexcept(std::forward<MoveElemFunctor>(moveElem)(std::move(std::declval<T>()))) && noexcept(spin(false))) {
        QueueResult ret
            = popIndex(nextTailOpt(), std::forward<MoveElemFunctor>(moveElem), std::forward<SpinFunctor>(spin));
        if (ret == QueueResult::Success) {
            pushPopSpin_.elementPopped();
        }
        return ret;
    }

    /* Calls `popWithSpinFunctor` with the default spin functor specified through
     * template argument of the LowLatencyQueue. Default is a busy-wait.
     *
     * @param moveElem  Functor that gets called with an rvalue of the element to
     * be poped. The user can decide what to do.
     * @return          QueueResult (self-describing). For optimistic approaches
     * QueueResult::SpuriousFailure is not expected.
     */
    template <typename MoveElemFunctor>
    QueueResult pop(MoveElemFunctor&& moveElem) noexcept(
        noexcept(std::forward<MoveElemFunctor>(moveElem)(std::move(std::declval<T>())))
        && noexcept(pushPopSpin_.spinOnPop()(false))) {
        return popWithSpinFunctor(std::forward<MoveElemFunctor>(moveElem), pushPopSpin_.spinOnPop());
    }

    /* Calls `popWithSpinFunctor` with the default spin functor specified through
     * template argument of the LowLatencyQueue. Default is a busy-wait. Throws if
     * the queue has been aborted.
     *
     * @param moveElem  Functor that gets called with an rvalue of the element to
     * be poped. The user can decide what to do.
     * @return          QueueResult (self-describing). For optimistic approaches
     * QueueResult::SpuriousFailure is not expected.
     */
    template <typename MoveElemFunctor>
    QueueResult popOrThrow(MoveElemFunctor&& moveElem) {
        QueueResult ret = pop(std::forward<MoveElemFunctor>(moveElem));
        if (ret == QueueResult::Aborted) {
            throwIfAborted();
        }
        return ret;
    }

    /* Calls `popWithSpinFunctor` with the default spin functor specified through
     * template argument of the LowLatencyQueue. Default is a busy-wait.
     *
     * @return        Returns the poped value or nothing if the queue has been
     * aborted or closed. Spurious failures are not expected for optimistic
     * approach.
     */
    std::optional<T> popAsOpt() noexcept(noexcept(std::is_nothrow_move_constructible<T>::value)
                                         && noexcept(pushPopSpin_.spinOnPop()(false))) {
        std::optional<T> ret;
        if (popWithSpinFunctor(
                [&ret](T&& v) noexcept(noexcept(ret = std::move(v))) {
                    ret.emplace(std::move(v));  // Will call move contructor on
                                                // transition from nothing to something
                },
                pushPopSpin_.spinOnPop())
            == QueueResult::Success) {
            return ret;
        }
        return std::nullopt;
    }

    /* Calls `popWithSpinFunctor` with the default spin functor specified through
     * template argument of the LowLatencyQueue. Default is a busy-wait.
     *
     * @return        Returns the poped value or nothing if the queue has been
     * aborted or closed. Spurious failures are not expected for optimistic
     * approach.
     */
    std::optional<T> popAsOptOrThrow() {
        std::optional<T> ret = popAsOpt();
        if (!ret) {
            throwIfAborted();
        }
        return ret;
    }

    /**
     * Pessimistic enqueu: First checks the size of the queue and tries to acquire
     * an entry location without oversubscribing. If it is not possible to acquire
     * an entry immediately, the function returns `false` and the user can decide
     * to retry or do something else.
     *
     * Once an entry location has been acquired, the thread tries to acquire the
     * entry state and to construct a value. If both approaches (push and tryPush)
     * are used in combination, in the case of high-contention the pessimistic
     * approach may fail very often although the buffer is not at it's capacity.
     * That's because an optimistic FAA instruction (fetch and add) will always
     * succeed and make concurrent CAS instruction (compare and swap) fail.
     *
     * Notes:
     *   * When you end up having a loop just calling tryPush until it succeeds,
     * it is better to use the optimistic `push`.
     *   * The SpinFunctor is only used when the rare case occurs (if at all...)
     * that two producers content for the same entry or that an entry is still
     * about to be poped.
     *
     * @param spin         Function that is called everytime the thread has to
     * spin and wait for an state change of an entry.
     * @param elemOrArgs   Forwards all arguments to the constructor of the
     * element type T. (May perform a copy or move construction or just emplaces a
     * value).
     * @return             QueueResult (self-describing).
     */
    template <typename SpinFunctor, typename... ElemOrArgs>
    QueueResult tryPushWithSpinFunctor(SpinFunctor&& spin, ElemOrArgs&&... elemOrArgs) noexcept(
        noexcept(T(std::forward<ElemOrArgs>(elemOrArgs)...)) && noexcept(spin(false))) {
        if (close_.load(std::memory_order_relaxed)) {
            return QueueResult::Closed;
        }
        auto indt = nextHeadPes();
        if (indt) {
            pushIndex(indt, std::forward<SpinFunctor>(spin), std::forward<ElemOrArgs>(elemOrArgs)...);
            pushPopSpin_.elementPushed();
            return QueueResult::Success;
        }
        return QueueResult::SpuriousFailure;
    }

    /* Calls `tryPushWithSpinFunctor` with the default spin functor specified
     * through template argument of the LowLatencyQueue. Default is a busy-wait.
     *
     * @param elemOrArgs   Forwards all arguments to the constructor of the
     * element type T. (May perform a copy or move construction or just emplaces a
     * value).
     * @return             QueueResult (self-describing).
     */
    template <typename... ElemOrArgs>
    QueueResult tryPush(ElemOrArgs&&... elemOrArgs) noexcept(
        noexcept(tryPushWithSpinFunctor(pushPopSpin_.spinOnPush(), std::forward<ElemOrArgs>(elemOrArgs)...))) {
        return tryPushWithSpinFunctor(pushPopSpin_.spinOnPush(), std::forward<ElemOrArgs>(elemOrArgs)...);
    }

    /**
     * Pessimistic pop: First checks the size of the queue and tries to acquire an
     * entry location without oversubscribing. If it is not possible to acquire an
     * entry immediately, the function returns `false` and the user can decide to
     * retry or do something else.
     *
     * Once an entry location has been acquired, the thread tries to acquire the
     * entry state and to construct a value. If both approaches (push and tryPush)
     * are used in combination, in the case of high-contention the pessimistic
     * approach may fail very often although the buffer is not at it's capacity.
     * That's because an optimistic FAA instruction (fetch and add) will always
     * succeed and make concurrent CAS instruction (compare and swap) fail.
     *
     *
     * Notes:
     *   * When you end up having a loop just calling tryPop until it succeeds, it
     * is better to use the optimistic `pop`.
     *   * The SpinFunctor is only used when the rare case occurs (if at all...)
     * that two consumers content for the same entry or that an entry is still
     * about to be pushed.
     *
     * @param moveElem  Functor that gets called with an rvalue of the element to
     * be poped. The user can decide what to do.
     * @param spin      Function that is called everytime the thread has to spin
     * and wait for an state change of an entry.
     * @return          QueueResult (self-describing).
     */
    template <typename MoveElemFunctor, typename SpinFunctor>
    QueueResult tryPopWithSpinFunctor(MoveElemFunctor&& moveElem, SpinFunctor&& spin) noexcept(
        noexcept(std::forward<MoveElemFunctor>(moveElem)(std::move(std::declval<T>()))) && noexcept(spin(false))) {
        auto indt = nextTailPes();
        if (indt) {
            QueueResult ret = popIndex(indt, std::forward<MoveElemFunctor>(moveElem), std::forward<SpinFunctor>(spin));
            if (ret == QueueResult::Success) {
                pushPopSpin_.elementPopped();
            }
            return ret;
        }
        return QueueResult::SpuriousFailure;
    }

    /* Calls `tryPopWithSpinFunctor` with the default spin functor specified
     * through template argument of the LowLatencyQueue. Default is a busy-wait.
     *
     * @param moveElem  Functor that gets called with an rvalue of the element to
     * be poped. The user can decide what to do.
     * @return          QueueResult (self-describing).
     */
    template <typename MoveElemFunctor>
    QueueResult tryPop(MoveElemFunctor&& moveElem) noexcept(
        noexcept(std::forward<MoveElemFunctor>(moveElem)(std::move(std::declval<T>())))
        && noexcept(pushPopSpin_.spinOnPop()(false))) {
        return tryPopWithSpinFunctor(std::forward<MoveElemFunctor>(moveElem), pushPopSpin_.spinOnPop());
    }

    /* Calls `tryPopWithSpinFunctor` with the default spin functor specified
     * through template argument of the LowLatencyQueue. Default is a busy-wait.
     *
     * @return Returns an optional that either contains a value when the pop
     * operation was successful.
     */
    std::optional<T> tryPopAsOpt() noexcept(noexcept(std::is_nothrow_move_constructible<T>::value)
                                            && noexcept(pushPopSpin_.spinOnPop()(false))) {
        std::optional<T> ret;
        if (tryPopWithSpinFunctor(
                [&ret](T&& v) noexcept(noexcept(ret = std::move(v))) {
                    ret.emplace(std::move(v));  // Will call move contructor
                },
                pushPopSpin_.spinOnPop())) {
            return ret;
        }
        return std::nullopt;
    }

    /* Calls `tryPopWithSpinFunctor` with the default spin functor specified
     * through template argument of the LowLatencyQueue. Default is a busy-wait.
     * Throws if the queue has been aborted.
     *
     * @return Returns an optional that either contains a value when the pop
     * operation was successful.
     */
    std::optional<T> tryPopAsOptOrThrow() {
        std::optional<T> ret = tryPopAsOpt();
        if (!ret) {
            throwIfAborted();
        }
        return ret;
    }
};

}  // namespace details

template <typename T, class AllocatorOrSize = AlignedAllocator<T, CACHELINE_SIZE_DESTRUCTIVE>,
          typename PushPopSpinFunctor = DefaultPushPopSpinFunctor<>, typename INDT = unsigned int>
class LowLatencyQueue : public details::LowLatencyQueueBase<T, AllocatorOrSize, PushPopSpinFunctor, INDT> {
private:
    using Base = details::LowLatencyQueueBase<T, AllocatorOrSize, PushPopSpinFunctor, INDT>;

public:
    LowLatencyQueue(const LowLatencyQueue&) = delete;

    LowLatencyQueue(LowLatencyQueue&& other) noexcept : Base(std::move(other)) {};

    LowLatencyQueue& operator=(const LowLatencyQueue& other) = delete;

    LowLatencyQueue& operator=(LowLatencyQueue&& other) noexcept { Base::operator=(std::move(other)); };

    LowLatencyQueue(std::size_t size) noexcept(noexcept(Base(size))) : Base(size) {};
};

template <typename T, typename IntType, IntType N, typename PushPopSpinFunctor, typename INDT>
struct LowLatencyQueue<T, std::integral_constant<IntType, N>, PushPopSpinFunctor, INDT>
    : public details::LowLatencyQueueBase<T, std::integral_constant<IntType, N>, PushPopSpinFunctor, INDT> {
private:
    using Base = details::LowLatencyQueueBase<T, std::integral_constant<IntType, N>, PushPopSpinFunctor, INDT>;

public:
    LowLatencyQueue(const LowLatencyQueue&) = delete;

    LowLatencyQueue(LowLatencyQueue&& other) noexcept : Base(std::move(other)) {};

    LowLatencyQueue& operator=(const LowLatencyQueue& other) = delete;

    LowLatencyQueue& operator=(LowLatencyQueue&& other) noexcept { Base::operator=(std::move(other)); };

    LowLatencyQueue() noexcept(noexcept(Base())) : Base() {};
};

}  // namespace llq

#endif

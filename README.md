# C++ Low-Latency queue 

Implementation of a fixed-size circular buffer that uses atomic states on each entry. 
The head and tail counters are optimistically incremented with fast fetch-and-add (FAA), which is mmuch faster than compare-and-swap, especially under high contention. 
The counters are then mapped to index/entry in the queue where synchronization happens on a local state (i.e. contentiont is distributed to specific entries). 

The streghth of the design is expected to work with optimistic push & pop instructions - e.g. no `try` attempts.

By avoiding resizing and memory allocation, a lot of typical problems like the ABA problem are avoided completly. 
If an application is designed properly, often it is enough to introduce more queue for different purposes and to choose their size properly.
The amount of consumers & producers always needs to be equilibrated and a queue should be large enough to absorb burst of pushes. 


# More details & techniques...

* optimistic pushs & pop: Because head and tail are incremented directly with an atomic FAA, it is not possible to make checks like "is the queue ful?". 
  If the number of pushs is higher then the size of the queue, the last `pushs` will always be blocking/spinning - likewise a `pop` on a empty queue is also blocking/spinning.
  
* Customizable `SpinFunctor` to customize spinning behaviour: 
    * i.e. introducing kernel calls (yielding, sleeping, condition variable)
    
* shuffled mapping from counter to an index/entry in the queue to avoid false sharing: 
  In a parallel environment it is typically assumed that multiple consecutive accesses (push/pop) happen from different producers/consumers.
  E.g. 4 produces want to push and the fetch-and-add results on head counters 0, 1, 2 and 3 - now for a queue of size 128, accesses are performed on entries
  0, 32, 64 and 96 which should all relate to different cache lines instead.
  

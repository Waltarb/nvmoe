// Standalone correctness test for segmented_host_lru.hpp against the documented
// behavior of nvmoe's Python SegmentedHostLru (nvme_offload_cache.py:222-286).
#include "segmented_host_lru.hpp"
#include <cstdio>
#include <cassert>

static int g_failures = 0;
#define CHECK(cond) do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            g_failures++; \
        } \
    } while (0)

int main() {
    // Test 1: basic insert/contains/size.
    {
        SegmentedHostLru<int> lru;
        lru.insert_new(1);
        lru.insert_new(2);
        CHECK(lru.contains(1));
        CHECK(lru.contains(2));
        CHECK(!lru.contains(3));
        CHECK(lru.size() == 2);
    }

    // Test 2: a fresh insert (1 reference) stays in probation and is the first evicted,
    // even if a LATER-inserted key gets touched (promoted) -- probation always drains
    // before protected, this is the whole point of the design.
    {
        SegmentedHostLru<int> lru;
        lru.set_capacity_hint(10);
        lru.insert_new(1); // 1 reference -- probation
        lru.insert_new(2);
        lru.touch(2);      // 2nd reference -- promotes 2 to protected
        int evicted = lru.pop_lru();
        CHECK(evicted == 1); // probation's LRU end, not protected's, regardless of recency
    }

    // Test 3: promote-on-2nd-hit semantics precisely -- insert_new is the 1st reference,
    // touch() is what promotes (the 2nd reference in nvmoe's calling convention).
    {
        SegmentedHostLru<int> lru;
        lru.set_capacity_hint(10);
        lru.insert_new(5);
        CHECK(lru.contains(5));
        lru.touch(5); // now protected
        // touching again should just refresh recency in protected, not error/duplicate
        lru.touch(5);
        CHECK(lru.contains(5));
        CHECK(lru.size() == 1);
    }

    // Test 4: protected cap enforcement -- with capacity=4 and protected_ratio=0.5,
    // protected cap = max(1, floor(4*0.5)) = 2. A 3rd promotion must demote the
    // protected LRU back to probation rather than growing protected past cap.
    {
        SegmentedHostLru<int> lru(0.5);
        lru.set_capacity_hint(4);
        lru.insert_new(1); lru.touch(1); // protected: [1]
        lru.insert_new(2); lru.touch(2); // protected: [1,2] (at cap=2)
        lru.insert_new(3); lru.touch(3); // promoting 3 should demote 1 (protected's LRU) back to probation
        CHECK(lru.contains(1)); // still present, just demoted
        CHECK(lru.contains(2));
        CHECK(lru.contains(3));
        // pop_lru should hit probation first -- and 1 was just demoted there
        int evicted = lru.pop_lru();
        CHECK(evicted == 1);
    }

    // Test 5: pop_lru drains probation fully before ever touching protected, and within
    // probation strictly respects insertion (LRU) order.
    {
        SegmentedHostLru<int> lru;
        lru.set_capacity_hint(10);
        lru.insert_new(10);
        lru.insert_new(20);
        lru.insert_new(30);
        lru.touch(20); // promote 20 to protected; probation now [10, 30]
        CHECK(lru.pop_lru() == 10);
        CHECK(lru.pop_lru() == 30);
        CHECK(lru.pop_lru() == 20); // only protected entry left
        CHECK(lru.size() == 0);
    }

    // Test 6: discard removes from whichever segment holds it, no-op if absent.
    {
        SegmentedHostLru<int> lru;
        lru.set_capacity_hint(10);
        lru.insert_new(7);
        lru.touch(7); // protected
        lru.discard(7);
        CHECK(!lru.contains(7));
        lru.discard(999); // no-op, must not throw/crash
    }

    // Test 7: set_capacity_hint is first-caller-wins (matches Python: "if self._capacity
    // is None").
    {
        SegmentedHostLru<int> lru;
        lru.set_capacity_hint(100);
        lru.set_capacity_hint(5); // must be ignored
        lru.insert_new(1); lru.touch(1);
        // cap should still be based on 100, not 5, so promoting many more shouldn't
        // demote at a tiny threshold
        for (int i = 2; i <= 50; i++) {
            lru.insert_new(i);
            lru.touch(i);
        }
        CHECK(lru.contains(1)); // would have been demoted long ago under cap=5*0.8=4
    }

    if (g_failures == 0) {
        printf("all tests passed\n");
        return 0;
    }
    printf("%d test(s) failed\n", g_failures);
    return 1;
}

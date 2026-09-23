// Phase 4, step 1: C++ port of nvmoe's SegmentedHostLru (nvme_offload_cache.py) --
// a 2Q/SLRU policy for the host RAM tier. Naive single-queue LRU evicts a recurring
// expert the instant it's least-recently-used even if it already proved itself worth a
// second reference; nvmoe's own live measurement found ~7.2% of dynamic-tier evictions
// under naive LRU had 2+ prior hits. This splits the dynamic pool into a probationary
// queue (first-time misses) and a protected queue (promoted on a 2nd reference), capped
// at `protected_ratio` of total capacity so protected can't swallow the whole pool and
// starve eviction candidates. Eviction always drains probation's LRU end first, falling
// back to protected's LRU end only once probation is empty -- so a genuine one-hit
// wonder still gets evicted quickly, it just can't take a proven-recurring expert down
// with it.
//
// This is a faithful behavioral port of the Python OrderedDict-based version: every
// method here has an O(1) counterpart there (move_to_end, popitem(last=False), pop,
// __contains__, __setitem__), implemented here as an intrusive doubly-linked list plus
// a hash map of iterators, the standard O(1) LRU-cache construction.
#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <optional>

template <typename Key>
class OrderedKeySet {
public:
    bool contains(const Key & k) const { return index_.count(k) != 0; }
    size_t size() const { return order_.size(); }
    bool empty() const { return order_.empty(); }

    // Insert at the MRU (back) end. Caller must ensure key is not already present.
    void push_back(const Key & k) {
        order_.push_back(k);
        auto it = order_.end();
        --it;
        index_[k] = it;
    }

    // Move an existing key to the MRU (back) end.
    void move_to_back(const Key & k) {
        auto found = index_.find(k);
        if (found == index_.end()) {
            return;
        }
        order_.splice(order_.end(), order_, found->second);
    }

    // Remove and return the LRU (front) key. Caller must ensure non-empty.
    Key pop_front() {
        Key k = order_.front();
        order_.pop_front();
        index_.erase(k);
        return k;
    }

    template <typename Predicate>
    std::optional<Key> pop_first_matching(Predicate pred) {
        for (auto it = order_.begin(); it != order_.end(); ++it) {
            if (pred(*it)) {
                Key k = *it;
                index_.erase(k);
                order_.erase(it);
                return k;
            }
        }
        return std::nullopt;
    }

    void erase(const Key & k) {
        auto found = index_.find(k);
        if (found == index_.end()) {
            return;
        }
        order_.erase(found->second);
        index_.erase(found);
    }

private:
    std::list<Key> order_;
    std::unordered_map<Key, typename std::list<Key>::iterator> index_;
};

template <typename Key>
class SegmentedHostLru {
public:
    explicit SegmentedHostLru(double protected_ratio = 0.8) : protected_ratio_(protected_ratio) {}

    bool contains(const Key & k) const { return probation_.contains(k) || protected_.contains(k); }
    size_t size() const { return probation_.size() + protected_.size(); }

    // Fix total dynamic-tier slot count once known -- constant after startup pinning
    // completes, so only needs setting once (first caller wins).
    void set_capacity_hint(int64_t capacity) {
        if (!capacity_.has_value()) {
            capacity_ = capacity;
        }
    }

    void insert_new(const Key & k) {
        probation_.push_back(k);
    }

    // Record a hit: promote probation -> protected on the 2nd reference; a key already
    // in protected just gets its recency refreshed there.
    void touch(const Key & k) {
        if (protected_.contains(k)) {
            protected_.move_to_back(k);
            return;
        }
        if (probation_.contains(k)) {
            probation_.erase(k);
            int64_t cap = protected_cap();
            if (cap >= 0 && (int64_t) protected_.size() >= cap && !protected_.empty()) {
                Key demoted = protected_.pop_front();
                probation_.push_back(demoted);
            }
            protected_.push_back(k);
        }
    }

    Key pop_lru() {
        if (!probation_.empty()) {
            return probation_.pop_front();
        }
        return protected_.pop_front();
    }

    template <typename Predicate>
    Key pop_lru_matching(Predicate pred) {
        auto cand = probation_.pop_first_matching(pred);
        if (cand.has_value()) {
            return *cand;
        }
        cand = protected_.pop_first_matching(pred);
        if (cand.has_value()) {
            return *cand;
        }
        return pop_lru();
    }

    void discard(const Key & k) {
        probation_.erase(k);
        protected_.erase(k);
    }

private:
    int64_t protected_cap() const {
        if (!capacity_.has_value()) {
            return -1; // unset -- no cap yet, matches Python's None-capacity behavior
        }
        return std::max<int64_t>(1, (int64_t) (*capacity_ * protected_ratio_));
    }

    OrderedKeySet<Key> probation_;
    OrderedKeySet<Key> protected_;
    double protected_ratio_;
    std::optional<int64_t> capacity_;
};

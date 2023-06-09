/*
 * Copyright (c) 2020 Nobuyuki Umetani
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

namespace stealth::internal {

template<typename T, typename Func>
inline void parallel_for(
        T num,
        Func &&func,
        unsigned int nthreads) {
    auto futures = std::vector<std::future<void>>{};

    std::atomic<T> next_idx(0);
    std::atomic<bool> has_error(false);
    for (auto thread_id = 0; thread_id < (int) nthreads; thread_id++) {
        futures.emplace_back(
                std::async(std::launch::async, [&func, &next_idx, &has_error, num, thread_id]() {
                    try {
                        while (true) {
                            auto idx = next_idx.fetch_add(1);
                            if (idx >= num) break;
                            if (has_error) break;
                            func(idx, thread_id);
                        }
                    } catch (...) {
                        has_error = true;
                        throw;
                    }
                }));
    }
    for (auto &f: futures) f.get();
}

template<typename T, typename Func>
inline void parallel_for(
        T num1,
        T num2,
        Func &&func,
        unsigned int nthreads) {
    auto futures = std::vector<std::future<void>>{};
    std::atomic<T> next_idx(0);
    std::atomic<bool> has_error(false);
    for (auto thread_id = 0; thread_id < (int) nthreads; thread_id++) {
        futures.emplace_back(std::async(
                std::launch::async, [&func, &next_idx, &has_error, num1, num2, thread_id]() {
                    try {
                        while (true) {
                            auto j = next_idx.fetch_add(1);
                            if (j >= num2) break;
                            if (has_error) break;
                            for (auto i = (T) 0; i < num1; i++) func(i, j, thread_id);
                        }
                    } catch (...) {
                        has_error = true;
                        throw;
                    }
                }));
    }
    for (auto &f: futures) f.get();
}

} // namespace stealth
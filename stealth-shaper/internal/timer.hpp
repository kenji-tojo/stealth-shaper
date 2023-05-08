#pragma once

#include <chrono>


namespace stealth::internal {

class Timer {
public:
    Timer() { restart(); }
    void restart() { m_start = std::chrono::high_resolution_clock::now(); }
    void stop() { m_end = std::chrono::high_resolution_clock::now(); }

    [[nodiscard]] double elapsed_sec() const {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(m_end-m_start);
        return static_cast<double>(ms.count()) / 1000.;
    }

private:
    using time_point_t = std::chrono::time_point<std::chrono::high_resolution_clock>;
    time_point_t m_start;
    time_point_t m_end;

};

} // namespace stealth::internal
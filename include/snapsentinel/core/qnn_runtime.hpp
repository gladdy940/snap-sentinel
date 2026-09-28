#pragma once

namespace snapsentinel::core {

class QnnRuntime final {
public:
    QnnRuntime() = default;
    ~QnnRuntime();

    QnnRuntime(const QnnRuntime&) = delete;
    QnnRuntime& operator=(const QnnRuntime&) = delete;
    QnnRuntime(QnnRuntime&& other) noexcept;
    QnnRuntime& operator=(QnnRuntime&& other) noexcept;

    void load();
    [[nodiscard]] bool isLoaded() const noexcept;

private:
    void* systemLibrary_ = nullptr;
    void* htpLibrary_ = nullptr;
};

}  // namespace snapsentinel::core

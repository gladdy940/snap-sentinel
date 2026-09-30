#include "qnn_runtime.hpp"

#include <dlfcn.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace snapsentinel::platform::android::qnn {
namespace {

void* openLibrary(const char* libraryName) {
    dlerror();
    void* handle = dlopen(libraryName, RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error(
            std::string("Unable to load ") + libraryName + ": " +
            (error != nullptr ? error : "unknown dynamic loader error")
        );
    }
    return handle;
}

}  // namespace

QnnRuntime::~QnnRuntime() {
    if (htpLibrary_ != nullptr) {
        dlclose(htpLibrary_);
    }
    if (systemLibrary_ != nullptr) {
        dlclose(systemLibrary_);
    }
}

QnnRuntime::QnnRuntime(QnnRuntime&& other) noexcept
    : systemLibrary_(std::exchange(other.systemLibrary_, nullptr)),
      htpLibrary_(std::exchange(other.htpLibrary_, nullptr)) {}

QnnRuntime& QnnRuntime::operator=(QnnRuntime&& other) noexcept {
    if (this != &other) {
        if (htpLibrary_ != nullptr) {
            dlclose(htpLibrary_);
        }
        if (systemLibrary_ != nullptr) {
            dlclose(systemLibrary_);
        }
        systemLibrary_ = std::exchange(other.systemLibrary_, nullptr);
        htpLibrary_ = std::exchange(other.htpLibrary_, nullptr);
    }
    return *this;
}

void QnnRuntime::load() {
    if (isLoaded()) {
        return;
    }

    void* systemLibrary = openLibrary("libQnnSystem.so");
    try {
        htpLibrary_ = openLibrary("libQnnHtp.so");
    } catch (...) {
        dlclose(systemLibrary);
        throw;
    }
    systemLibrary_ = systemLibrary;
}

bool QnnRuntime::isLoaded() const noexcept {
    return systemLibrary_ != nullptr && htpLibrary_ != nullptr;
}

}  // namespace snapsentinel::platform::android::qnn

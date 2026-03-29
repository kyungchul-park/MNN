//
//  TraceUtils.hpp
//  MNN
//
//  Lightweight Android ATrace helpers used by optional profiling code.
//

#ifndef MNN_TraceUtils_hpp
#define MNN_TraceUtils_hpp

#include <cstdlib>
#include <string>

#ifdef __ANDROID__
#include <android/trace.h>
#endif

namespace MNN {

inline bool envFlagEnabled(const char* key, bool defaultValue = false) {
    auto value = std::getenv(key);
    if (nullptr == value) {
        return defaultValue;
    }
    if (value[0] == '1' || value[0] == 'y' || value[0] == 'Y' || value[0] == 't' || value[0] == 'T') {
        return true;
    }
    if (value[0] == '0' || value[0] == 'n' || value[0] == 'N' || value[0] == 'f' || value[0] == 'F') {
        return false;
    }
    return defaultValue;
}

inline bool openCLTraceEnabled() {
    static const bool enabled = envFlagEnabled("MNN_OPENCL_PROFILE") || envFlagEnabled("MNN_OPENCL_ATRACE");
    return enabled;
}

inline bool openCLLogEnabled() {
    static const bool enabled = envFlagEnabled("MNN_OPENCL_PROFILE") || envFlagEnabled("MNN_OPENCL_PROFILE_LOG");
    return enabled;
}

inline void traceBegin(const char* name) {
#ifdef __ANDROID__
    if (openCLTraceEnabled() && ATrace_isEnabled()) {
        ATrace_beginSection(name);
    }
#else
    (void)name;
#endif
}

inline void traceEnd() {
#ifdef __ANDROID__
    if (openCLTraceEnabled() && ATrace_isEnabled()) {
        ATrace_endSection();
    }
#endif
}

class ScopedTrace {
public:
    ScopedTrace() = default;
    explicit ScopedTrace(const std::string& name) {
        begin(name);
    }
    explicit ScopedTrace(const char* name) {
        begin(name);
    }
    ~ScopedTrace() {
        if (mActive) {
            traceEnd();
        }
    }

    void begin(const std::string& name) {
        begin(name.c_str());
    }
    void begin(const char* name) {
        if (nullptr == name) {
            return;
        }
        traceBegin(name);
        mActive = openCLTraceEnabled();
    }

private:
    bool mActive = false;
};

} // namespace MNN

#endif

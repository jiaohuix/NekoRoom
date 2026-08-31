// MoeChat JNI bridge: minimal MNN LLM dialog
// Loads LLM via config_path → response(prompt) streams tokens through a Kotlin callback.
#include <android/log.h>
#include <jni.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include "llm/llm.hpp"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "MoeAvatarLLM", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "MoeAvatarLLM", __VA_ARGS__)

using MNN::Transformer::Llm;

namespace {

std::atomic<bool> g_perf_logging{false};

// streambuf that flushes each chunk into a callback (UTF-8 token bytes)
class CallbackStreamBuf : public std::streambuf {
public:
    using Cb = std::function<bool(const std::string&)>;  // return true to stop
    explicit CallbackStreamBuf(Cb cb) : cb_(std::move(cb)) {}
    bool stopped() const { return stopped_; }
protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        if (n > 0 && cb_) {
            stopped_ = stopped_ || cb_(std::string(s, static_cast<size_t>(n)));
        }
        return n;
    }
private:
    Cb cb_;
    bool stopped_ = false;
};

struct Session {
    std::unique_ptr<Llm> llm;
    std::atomic<bool> stop_flag{false};
};

inline Session* asSession(jlong ptr) {
    return reinterpret_cast<Session*>(ptr);
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_moeavatar_llm_LocalLlmBridge_initNative(JNIEnv* env, jobject /*thiz*/, jstring jConfigPath) {
    const char* cfg = env->GetStringUTFChars(jConfigPath, nullptr);
    LOGI("initNative configPath=%s", cfg);
    auto* session = new Session();
    session->llm.reset(Llm::createLLM(cfg));
    env->ReleaseStringUTFChars(jConfigPath, cfg);
    if (!session->llm) {
        LOGE("createLLM returned null");
        delete session;
        return 0;
    }
    if (!session->llm->load()) {
        LOGE("Llm::load() failed");
        delete session;
        return 0;
    }
    LOGI("Llm load OK ptr=%p", session);
    return reinterpret_cast<jlong>(session);
}

JNIEXPORT void JNICALL
Java_com_moeavatar_llm_LocalLlmBridge_resetNative(JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    auto* session = asSession(ptr);
    if (session && session->llm) {
        session->llm->reset();
    }
}

JNIEXPORT void JNICALL
Java_com_moeavatar_llm_LocalLlmBridge_stopNative(JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    auto* session = asSession(ptr);
    if (session) {
        session->stop_flag.store(true);
    }
}

JNIEXPORT void JNICALL
Java_com_moeavatar_llm_LocalLlmBridge_releaseNative(JNIEnv* /*env*/, jobject /*thiz*/, jlong ptr) {
    auto* session = asSession(ptr);
    delete session;
}

JNIEXPORT void JNICALL
Java_com_moeavatar_llm_LocalLlmBridge_setPerfLoggingNative(JNIEnv* /*env*/, jobject /*thiz*/, jboolean enabled) {
    g_perf_logging.store(enabled == JNI_TRUE);
    LOGI("performance logging %s", enabled == JNI_TRUE ? "enabled" : "disabled");
}

JNIEXPORT jstring JNICALL
Java_com_moeavatar_llm_LocalLlmBridge_submitNative(JNIEnv* env, jobject /*thiz*/,
                                        jlong ptr, jstring jPrompt, jobject jListener) {
    auto* session = asSession(ptr);
    if (!session || !session->llm) {
        return env->NewStringUTF("");
    }
    session->stop_flag.store(false);
    const auto t_start = std::chrono::steady_clock::now();
    bool first_flushed = false;

    const char* prompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string promptStr(prompt);
    env->ReleaseStringUTFChars(jPrompt, prompt);

    // Resolve listener.onToken(String) once
    jclass listenerCls = env->GetObjectClass(jListener);
    jmethodID onTokenMid = env->GetMethodID(listenerCls, "onToken", "(Ljava/lang/String;)Z");
    if (onTokenMid == nullptr) {
        LOGE("listener.onToken(String):Z not found");
        env->DeleteLocalRef(listenerCls);
        return env->NewStringUTF("");
    }

    // Find length in bytes of the longest UTF-8-complete prefix of [data, data+n).
    // Tokens may split a multi-byte char; flushing a half char to NewStringUTF aborts the JVM.
    auto utf8CompletePrefix = [](const char* data, size_t n) -> size_t {
        if (n == 0) return 0;
        size_t i = n;
        // walk back over continuation bytes (10xxxxxx)
        while (i > 0 && (static_cast<unsigned char>(data[i - 1]) & 0xC0) == 0x80) {
            --i;
        }
        if (i == 0) return 0;
        unsigned char lead = static_cast<unsigned char>(data[i - 1]);
        size_t need = 1;
        if      ((lead & 0x80) == 0x00) need = 1;
        else if ((lead & 0xE0) == 0xC0) need = 2;
        else if ((lead & 0xF0) == 0xE0) need = 3;
        else if ((lead & 0xF8) == 0xF0) need = 4;
        else return i - 1;  // malformed lead, drop it
        size_t have = n - (i - 1);
        return (have >= need) ? n : (i - 1);
    };

    std::string pending;
    std::stringstream allText;
    auto flush = [&](const std::string& s) -> bool {
        if (s.empty()) return false;
        if (g_perf_logging.load() && !first_flushed) {
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t_start).count();
            LOGI("perf stage=llm_first_chunk ms=%lld prompt_bytes=%zu",
                 static_cast<long long>(ms), promptStr.size());
            first_flushed = true;
        }
        jstring js = env->NewStringUTF(s.c_str());
        jboolean stop = env->CallBooleanMethod(jListener, onTokenMid, js);
        env->DeleteLocalRef(js);
        if (env->ExceptionCheck()) {
            env->ExceptionDescribe();
            env->ExceptionClear();
            return true;
        }
        return stop == JNI_TRUE;
    };

    auto cb = [&](const std::string& chunk) -> bool {
        if (session->stop_flag.load()) return true;
        allText << chunk;
        pending.append(chunk);
        size_t cut = utf8CompletePrefix(pending.data(), pending.size());
        if (cut == 0) return false;  // wait for more bytes
        std::string emit(pending.data(), cut);
        pending.erase(0, cut);
        return flush(emit);
    };

    CallbackStreamBuf buf(cb);
    std::ostream os(&buf);
    try {
        session->llm->response(promptStr, &os, "<eop>", -1);
    } catch (const std::exception& e) {
        LOGE("response exception: %s", e.what());
    }
    // Flush any tail bytes (drop if still malformed at end).
    if (!pending.empty()) {
        size_t cut = utf8CompletePrefix(pending.data(), pending.size());
        if (cut > 0) flush(std::string(pending.data(), cut));
    }

    if (g_perf_logging.load()) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_start).count();
        LOGI("perf stage=llm_done ms=%lld output_bytes=%zu stopped=%d",
             static_cast<long long>(ms), allText.str().size(), session->stop_flag.load() ? 1 : 0);
    }

    env->DeleteLocalRef(listenerCls);
    return env->NewStringUTF(allText.str().c_str());
}

}  // extern "C"

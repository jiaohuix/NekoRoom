package com.moeavatar.llm

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.util.concurrent.atomic.AtomicLong

/**
 * 端侧 MNN-LLM 后端。给定一个 config.json 路径，load 后即可 chat。
 *
 * 历史拼接策略：用 ChatML 风格的简单拼接（user/assistant 各占一行 + role 前缀），
 * 末尾拼一个 "<|assistant|>\n" 让模型续写。MNN 的 Llm::response 会负责 KV 缓存。
 *
 * @property configPath 形如 /sdcard/.../<model_dir>/config.json
 */
class LocalLlmBackend(private val configPath: String) : LlmBackend {

    override val displayName: String = "本地 · ${File(configPath).parentFile?.name ?: "?"}"

    private val nativePtr = AtomicLong(0L)
    @Volatile private var prepared = false

    // stopNative 跨线程并发会 SIGSEGV：取消 chatJob 时 callbackFlow.awaitClose 会调一次，
    // onMicPressed 又主动调 backend.stop() 一次 —— 两次同时进 native 抢内部状态 → 闪退黑屏。
    // 锁只串行化 stop/reset/release；submitNative 本身是长阻塞调用，不能进锁，否则 stop 等不到它。
    // stopNative 设计上就该能跟 submitNative 并发执行（用来打断它）—— 这是 native 层 OK 的，
    // 同 stopNative 之间互锁就够。
    private val nativeLock = Any()

    override val ready: Boolean
        get() = prepared && nativePtr.get() != 0L

    override suspend fun prepare(): Boolean {
        if (ready) return true
        if (!LocalLlmBridge.nativeAvailable) {
            Log.w(TAG, "native lib not available, local backend disabled")
            return false
        }
        val ptr = try {
            LocalLlmBridge.initNative(configPath)
        } catch (t: Throwable) {
            Log.e(TAG, "initNative throw", t)
            0L
        }
        if (ptr == 0L) {
            Log.e(TAG, "initNative returned 0 for $configPath")
            return false
        }
        nativePtr.set(ptr)
        prepared = true
        return true
    }

    override fun chat(history: List<ChatTurn>): Flow<String> = callbackFlow<String> {
        val ptr = nativePtr.get()
        if (ptr == 0L) {
            close(IllegalStateException("local llm not prepared"))
            return@callbackFlow
        }
        val prompt = composePrompt(history)

        var emitted = 0
        val listener = object : LocalLlmBridge.TokenListener {
            override fun onToken(token: String): Boolean {
                if (token == "<eop>") return true
                // trySend 失败（下游已取消）时返回 stop=true，让 native 端尽快收尾
                val ok = trySend(token).isSuccess
                if (!ok) return true
                emitted++
                // 长度上限：聊天场景不要长篇大论。到软上限就在句末收尾（避免 TTS 半句被切），
                // 到硬上限无条件停（防某些回复一直不落标点）。
                if (emitted >= HARD_MAX_TOKENS) return true
                if (emitted >= SOFT_MAX_TOKENS && token.isNotEmpty() && SENTENCE_END.contains(token.last())) return true
                return false
            }
        }

        // submitNative 是阻塞的——必须放到 IO，但 callbackFlow 本身就在协程里，
        // 我们靠 flowOn(Dispatchers.IO) 切线程。
        try {
            LocalLlmBridge.submitNative(ptr, prompt, listener)
        } catch (t: Throwable) {
            Log.e(TAG, "submitNative throw", t)
            close(t)
            return@callbackFlow
        }

        close()  // 模型说完，正常结束 Flow
        awaitClose {
            // 上游取消时通知 native 停止 —— 必须跟外部 stop()/release() 串行
            synchronized(nativeLock) {
                runCatching { LocalLlmBridge.stopNative(ptr) }
            }
        }
    }.flowOn(Dispatchers.IO)

    override fun stop() {
        val ptr = nativePtr.get()
        if (ptr != 0L) synchronized(nativeLock) { LocalLlmBridge.stopNative(ptr) }
    }

    override fun resetSession() {
        val ptr = nativePtr.get()
        if (ptr != 0L) synchronized(nativeLock) { LocalLlmBridge.resetNative(ptr) }
    }

    override fun release() {
        val ptr = nativePtr.getAndSet(0L)
        if (ptr != 0L) synchronized(nativeLock) {
            try { LocalLlmBridge.stopNative(ptr) } catch (_: Throwable) {}
            try { LocalLlmBridge.releaseNative(ptr) } catch (_: Throwable) {}
        }
        prepared = false
    }

    /** 把对话历史拼成单条 prompt。MNN 的 Llm::response 内部会处理 chat template。 */
    private fun composePrompt(history: List<ChatTurn>): String {
        // 取最后一条 user 作为本轮输入，前面的 history 当上下文（让 native 内部处理）
        // 这个简单实现：直接把最后一条 user 内容传过去；KV cache 由 native 维护。
        val last = history.lastOrNull { it.role == ChatTurn.Role.USER }
        return last?.content ?: ""
    }

    companion object {
        private const val TAG = "LocalLlmBackend"
        // 聊天回复长度上限：软上限到句末即收尾，硬上限兜底。约束在 App 层，随包走、改 config 不影响。
        // token 约等于 1 个汉字/词片，40~64 token ≈ 1~2 句短回答，适合聊天。
        private const val SOFT_MAX_TOKENS = 40
        private const val HARD_MAX_TOKENS = 64
        private val SENTENCE_END = setOf('。', '！', '？', '.', '!', '?', '~', '…')
    }
}

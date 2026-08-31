package com.moeavatar.tts

import android.content.Context
import android.util.Log
import com.example.supertonic.SuperTonicInfer
import com.moeavatar.model.ModelManager
import com.moeavatar.perf.PerformanceTrace
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.Dispatchers

/**
 * SuperTonic-Neko 本地 TTS backend。单一猫娘音色，模型从设备端 [modelDir] 读（不进包）。
 *
 * SuperTonic 单次合成有 ~6.97s 上限，超过会截断。上游 [SentenceSplitter] 已按 maxHard=40
 * 切句，但 40 个汉字念出来可能超 7s，所以这里再按 [MAX_CLAUSE_CHARS] 子切一次，逐段合成、
 * 逐段 emit，规避上限。
 */
class SuperTonicTtsBackend(
    context: Context,
    private val modelDir: String = ModelManager.ttsDir(context),
    private val voiceId: String = "catgirl_style",
) : TtsBackend {

    override val displayName: String = "SuperTonic-Neko"

    private val infer = SuperTonicInfer(context.applicationContext)
    @Volatile private var ready = false

    override suspend fun prepare(): Boolean {
        if (ready) return true
        val t0 = System.currentTimeMillis()
        infer.setPerformanceLogging(performanceLogging)
        ready = infer.init(modelDir, voiceId = voiceId)
        PerformanceTrace.i("tts_prepare", "ok=$ready ms=${System.currentTimeMillis() - t0}")
        if (!ready) Log.e(TAG, "SuperTonic init failed, modelDir=$modelDir")
        return ready
    }

    // --- 每轮回复的性能统计（首包延迟 + RTF）--------------------------------
    // firstPacketMs：本轮第一个 clause 的合成耗时（≈从有文本到出第一段音频的延迟）。
    // RTF = 累计合成耗时 / 累计音频时长。resetStats() 在每轮 LLM 回复开始时调。
    @Volatile private var firstPacketMs: Long = -1L
    @Volatile private var accSynthMs: Long = 0L
    @Volatile private var accAudioMs: Long = 0L

    fun resetStats() {
        firstPacketMs = -1L
        accSynthMs = 0L
        accAudioMs = 0L
    }

    fun setPerformanceLogging(enabled: Boolean) {
        performanceLogging = enabled
        if (ready) infer.setPerformanceLogging(enabled)
    }

    data class TtsStats(val firstPacketMs: Long, val rtf: Double)

    fun snapshotStats(): TtsStats {
        val rtf = if (accAudioMs > 0L) accSynthMs.toDouble() / accAudioMs else 0.0
        return TtsStats(firstPacketMs, rtf)
    }

    override fun synth(text: String, speaker: String, stopSignal: () -> Boolean): Flow<PcmChunk> = flow {
        if (!ready) return@flow
        val clauses = subSplit(text)
        for ((idx, clause) in clauses.withIndex()) {
            if (stopSignal()) return@flow
            val (pcm, ms) = infer.synth(clause)
            if (pcm == null || pcm.isEmpty()) {
                Log.w(TAG, "synth returned empty for: $clause")
                continue
            }
            if (firstPacketMs < 0L) firstPacketMs = ms
            accSynthMs += ms
            accAudioMs += pcm.size * 1000L / 44100
            Log.i(TAG, "clause '${clause}' -> ${pcm.size} samples in ${ms}ms (rtf=${infer.rtf(pcm, ms)})")
            PerformanceTrace.i("tts_clause", "index=$idx chars=${clause.length} samples=${pcm.size} inferMs=$ms rtf=${infer.rtf(pcm, ms)}")
            emit(PcmChunk(pcm, 44100, last = idx == clauses.lastIndex, text = clause))
        }
    }.flowOn(Dispatchers.Default)

    override fun release() {
        if (ready) {
            infer.release()
            ready = false
        }
    }

    /** 把一句话再按标点/长度切成 ≤MAX_CLAUSE_CHARS 的小段，优先在标点处断。 */
    private fun subSplit(text: String): List<String> {
        val t = text.trim()
        if (!t.any { it.isLetterOrDigit() }) return emptyList()  // 纯标点(如 "." "……")不合成，否则出 2.4s 填充噪声
        if (t.length <= MAX_CLAUSE_CHARS) return listOf(t)
        val out = mutableListOf<String>()
        val buf = StringBuilder()
        for (c in t) {
            buf.append(c)
            val atPunct = PUNCT.contains(c)
            if ((atPunct && buf.length >= MIN_CLAUSE_CHARS) || buf.length >= MAX_CLAUSE_CHARS) {
                out.add(buf.toString().trim())
                buf.setLength(0)
            }
        }
        if (buf.isNotBlank()) out.add(buf.toString().trim())
        return out.filter { it.any { ch -> ch.isLetterOrDigit() } }
    }

    companion object {
        @Volatile var performanceLogging: Boolean = false
        private const val TAG = "MoeAvatar.SuperTonic"
        private const val MIN_CLAUSE_CHARS = 8
        // 20 太小 → 24 字的整句被拦腰截成 "…想和我说个有趣"+"的事情吗？"。实测 ~0.23s/字，
        // 6.97s 合成上限约 30 字，取 28 留余量：既避免中途断词，又不触发截断。
        private const val MAX_CLAUSE_CHARS = 28
        private val PUNCT = setOf('。', '？', '！', '.', '?', '!', '，', '；', '、', ',', ';', ':', '：', '—', '\n')
    }
}

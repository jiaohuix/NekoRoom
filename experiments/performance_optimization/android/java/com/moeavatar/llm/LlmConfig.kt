package com.moeavatar.llm

import android.content.Context
import android.content.SharedPreferences
import androidx.core.content.edit
import com.moeavatar.model.ModelManager

/**
 * 全局可配置项。SharedPreferences 持久化。
 *
 * 这里只放跟 LLM/TTS 选择相关的轻量配置，模型扫描结果不持久化。
 */
class LlmConfig(ctx: Context) {

    private val appCtx: Context = ctx.applicationContext

    private val sp: SharedPreferences =
        appCtx.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    enum class BackendKind { LOCAL, OPENAI }

    var backendKind: BackendKind
        get() = runCatching { BackendKind.valueOf(sp.getString(K_BACKEND, BackendKind.LOCAL.name)!!) }
            .getOrDefault(BackendKind.LOCAL)
        set(v) = sp.edit { putString(K_BACKEND, v.name) }

    /** 本地模型扫描根目录，默认应用私有目录 models/llm（免存储权限） */
    var localModelDir: String
        get() = sp.getString(K_LOCAL_DIR, null) ?: ModelManager.llmScanRoot(appCtx)
        set(v) = sp.edit { putString(K_LOCAL_DIR, v) }

    /** 当前选中的本地模型子目录名（不是绝对路径） */
    var localModelName: String?
        get() = sp.getString(K_LOCAL_NAME, null)
        set(v) = sp.edit { putString(K_LOCAL_NAME, v) }

    var openAiBaseUrl: String
        get() = sp.getString(K_OAI_BASE, DEFAULT_OAI_BASE) ?: DEFAULT_OAI_BASE
        set(v) = sp.edit { putString(K_OAI_BASE, v) }

    var openAiApiKey: String
        get() = sp.getString(K_OAI_KEY, "") ?: ""
        set(v) = sp.edit { putString(K_OAI_KEY, v) }

    var openAiModel: String
        get() = sp.getString(K_OAI_MODEL, DEFAULT_OAI_MODEL) ?: DEFAULT_OAI_MODEL
        set(v) = sp.edit { putString(K_OAI_MODEL, v) }

    /**
     * v0.6.2 新增：当前选中的 Provider preset id（`deepseek`/`siliconflow`/`agnes`/`custom`）。
     * 老用户首次读到未设置时，按已存 openAiBaseUrl 一次性推断并写回。
     */
    var providerPresetId: String
        get() {
            val saved = sp.getString(K_OAI_PRESET, null)
            if (!saved.isNullOrBlank()) return saved
            val inferred = ProviderRegistry.inferByBaseUrl(openAiBaseUrl)
            sp.edit { putString(K_OAI_PRESET, inferred) }
            return inferred
        }
        set(v) = sp.edit { putString(K_OAI_PRESET, v) }

    /** v0.6.2 新增：温度参数，默认 0.7。UI 通过 SeekBar 调节 0.0–2.0。 */
    var temperature: Float
        get() = sp.getFloat(K_OAI_TEMP, 0.7f)
        set(v) = sp.edit { putFloat(K_OAI_TEMP, v) }

    /** v0.6.2 新增：是否启用思考模式。默认关闭（大幅降低首字延迟）。 */
    var enableThinking: Boolean
        get() = sp.getBoolean(K_OAI_THINKING, false)
        set(v) = sp.edit { putBoolean(K_OAI_THINKING, v) }

    var systemPrompt: String
        get() = sp.getString(K_SYS_PROMPT, DEFAULT_SYS_PROMPT) ?: DEFAULT_SYS_PROMPT
        set(v) = sp.edit { putString(K_SYS_PROMPT, v) }

    /** TTS 当前音色 id（沿用 BertVITS2 的 speaker 名字） */
    var ttsSpeaker: String
        get() = sp.getString(K_TTS_SPK, "甘雨_ZH") ?: "甘雨_ZH"
        set(v) = sp.edit { putString(K_TTS_SPK, v) }

    enum class TtsBackendKind { LOCAL, MINIMAX }

    /** TTS 后端：本地 BertVITS2 vs 在线 MiniMax 流式 */
    var ttsBackendKind: TtsBackendKind
        get() = runCatching { TtsBackendKind.valueOf(sp.getString(K_TTS_BACKEND, TtsBackendKind.LOCAL.name)!!) }
            .getOrDefault(TtsBackendKind.LOCAL)
        set(v) = sp.edit { putString(K_TTS_BACKEND, v.name) }

    /** MiniMax API Key（自己填，不写死） */
    var minimaxApiKey: String
        get() = sp.getString(K_MINIMAX_KEY, "") ?: ""
        set(v) = sp.edit { putString(K_MINIMAX_KEY, v) }

    /** MiniMax 音色 id，默认 female-shaonv */
    var minimaxVoiceId: String
        get() = sp.getString(K_MINIMAX_VOICE, "female-shaonv") ?: "female-shaonv"
        set(v) = sp.edit { putString(K_MINIMAX_VOICE, v) }

    /** 当前 Live2D 角色名（对应 Live2DController.PRESETS 的 key） */
    var live2dModel: String
        get() {
            val saved = sp.getString(K_LIVE2D, DEFAULT_LIVE2D) ?: DEFAULT_LIVE2D
            return if (saved in REMOVED_BUILTIN_MODELS) DEFAULT_LIVE2D else saved
        }
        set(v) = sp.edit { putString(K_LIVE2D, v) }

    /** 是否显示 AI 回复字幕（用户发送文本不受此开关影响，仍短暂显示） */
    var showAiSubtitle: Boolean
        get() = sp.getBoolean(K_SHOW_AI_SUB, true)
        set(v) = sp.edit { putBoolean(K_SHOW_AI_SUB, v) }

    /** 是否显示性能小字（首字延迟/tok/s、TTS RTF）。默认关。 */
    var showPerfLine: Boolean
        get() = sp.getBoolean(K_SHOW_PERF, false)
        set(v) = sp.edit { putBoolean(K_SHOW_PERF, v) }

    /**
     * dev 模式：默认关。打开后才显示二级性能优化选项。
     * - true: 所有 MoeAvatar.* TAG 输出 Verbose 级别
     * - false: 只输出 Warn 及以上
     *
     * 后续可以加 UI 开关；当前默认开便于排查电话模式/VAD 问题。
     */
    var devMode: Boolean
        get() = sp.getBoolean(K_DEV_MODE, false)
        set(v) = sp.edit { putBoolean(K_DEV_MODE, v) }

    /** Developer-only detailed stage timing logs to logcat. */
    var performanceLogsEnabled: Boolean
        get() = sp.getBoolean(K_PERF_LOGS, false)
        set(v) = sp.edit { putBoolean(K_PERF_LOGS, v) }

    /** Build-time SME2 preference. Changing this requires rebuilding libMNN.so. */
    var sme2BuildEnabled: Boolean
        get() = sp.getBoolean(K_SME2_BUILD, true)
        set(v) = sp.edit { putBoolean(K_SME2_BUILD, v) }

    /** 首启欢迎引导是否已展示过（只在第一次打开时弹能力安装引导）。 */
    var seenWelcome: Boolean
        get() = sp.getBoolean(K_SEEN_WELCOME, false)
        set(v) = sp.edit { putBoolean(K_SEEN_WELCOME, v) }

    /** AI 虚拟角色身份说明只在安装或升级后的首次进入展示一次。 */
    var seenAiDisclosure: Boolean
        get() = sp.getBoolean(K_SEEN_AI_DISCLOSURE, false)
        set(v) = sp.edit { putBoolean(K_SEEN_AI_DISCLOSURE, v) }

    /**
     * v0.5 Character Engine：是否允许 LLM 通过 <action>{...}</action> 块驱动
     * Live2D 表情/装扮。默认开；仅对在线后端生效，本地后端始终不注入。
     */
    var liveActingEnabled: Boolean
        get() = sp.getBoolean(K_LIVE_ACTING, true)
        set(v) = sp.edit { putBoolean(K_LIVE_ACTING, v) }

    companion object {
        private const val PREF_NAME = "moeavatar_llm"
        private const val K_BACKEND = "backend"
        private const val K_LOCAL_DIR = "local_dir"
        private const val K_LOCAL_NAME = "local_name"
        private const val K_OAI_BASE = "oai_base"
        private const val K_OAI_KEY = "oai_key"
        private const val K_OAI_MODEL = "oai_model"
        private const val K_OAI_PRESET = "oai_preset_id"
        private const val K_OAI_TEMP = "oai_temperature"
        private const val K_OAI_THINKING = "oai_enable_thinking"
        private const val K_SYS_PROMPT = "sys_prompt"
        private const val K_TTS_SPK = "tts_spk"
        private const val K_LIVE2D = "live2d_model"
        private const val K_TTS_BACKEND = "tts_backend"
        private const val K_MINIMAX_KEY = "minimax_key"
        private const val K_MINIMAX_VOICE = "minimax_voice"
        private const val K_DEV_MODE = "dev_mode"
        private const val K_SHOW_AI_SUB = "show_ai_sub"
        private const val K_SHOW_PERF = "show_perf"
        private const val K_PERF_LOGS = "performance_logs"
        private const val K_SME2_BUILD = "sme2_build_enabled"
        private const val K_SEEN_WELCOME = "seen_welcome"
        private const val K_SEEN_AI_DISCLOSURE = "seen_ai_disclosure_v1"
        private const val K_LIVE_ACTING = "live_acting"
        private const val DEFAULT_LIVE2D = "FenmaoLoli"
        private val REMOVED_BUILTIN_MODELS = setOf("Ziyan", "ATRI", "Yuuka", "Amadeus")

        const val DEFAULT_OAI_BASE = "https://api.deepseek.com/v1"
        const val DEFAULT_OAI_MODEL = "deepseek-v4-flash"
        const val DEFAULT_SYS_PROMPT =
            "你是一只可爱的猫娘，名字叫小喵。说话软萌、亲昵，偶尔在句尾加\"喵~\"。" +
            "回答简洁口语化，用中文聊天，不要使用 markdown、颜文字或长篇大论。"
    }
}

//
// SuperTonic-Neko on-device inference (MNN Express).
// See super_tonic_loader.hpp for the API contract.
//
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/NeuralNetWorkOp.hpp>
#include <algorithm>
#include <cmath>
#include <random>

#include "super_tonic_loader.hpp"

namespace SUPERTONIC_NEKO {

using namespace MNN::Express;

static bool g_initialized = false;
static Module::Config g_mdconfig;
static std::shared_ptr<Module> g_dp = nullptr;
static std::shared_ptr<Module> g_te = nullptr;
static std::shared_ptr<Module> g_ve = nullptr;
static std::shared_ptr<Module> g_voc = nullptr;
static std::vector<float> g_fixed_noise;  // empty = sample from seed
static bool g_perf_logging = false;

void set_perf_logging(bool enabled) {
    g_perf_logging = enabled;
    STNPRINT("performance logging %s\n", enabled ? "enabled" : "disabled");
}

#define STNPERF(fmt, ...) do { if (g_perf_logging) STNPRINT(fmt, ##__VA_ARGS__); } while (0)

template <typename T>
static T clampv(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

void init_loader(int num_threads) {
    if (g_initialized) {
        STNPRINT("loader already initialized\n");
        return;
    }
    MNN::BackendConfig bc;
    // Precision_Low lets MNN use fp16 kernels on ARM82 — matches our fp16 export
    // and is what keeps RTF ~= real-time on mobile.
    bc.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;
    bc.memory = MNN::BackendConfig::MemoryMode::Memory_Low;
    std::shared_ptr<Executor> exe = Executor::newExecutor(MNN_FORWARD_CPU, bc, num_threads);
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, bc, num_threads);
    ExecutorScope scope(exe);
    g_initialized = true;
    STNPRINT("loader initialized (threads=%d)\n", num_threads);
}

void destroy_loader() {
    g_dp.reset(); g_te.reset(); g_ve.reset(); g_voc.reset();
    g_fixed_noise.clear();
    g_initialized = false;
    STNPRINT("loader destroyed\n");
}

void set_model_path(const std::string& dp_mnn, const std::string& te_mnn,
                    const std::string& ve_mnn, const std::string& vocoder_mnn) {
    STNPRINT("loading models:\n  dp=%s\n  te=%s\n  ve=%s\n  voc=%s\n",
             dp_mnn.c_str(), te_mnn.c_str(), ve_mnn.c_str(), vocoder_mnn.c_str());

    const std::vector<std::string> dp_in{"text_ids", "style_dp", "text_mask"};
    const std::vector<std::string> dp_out{"duration"};
    g_dp.reset(Module::load(dp_in, dp_out, dp_mnn.c_str(), nullptr, &g_mdconfig));

    const std::vector<std::string> te_in{"text_ids", "pinyin_ids", "tone_ids",
                                         "prosody_ids", "style_ttl", "text_mask"};
    const std::vector<std::string> te_out{"text_emb"};
    g_te.reset(Module::load(te_in, te_out, te_mnn.c_str(), nullptr, &g_mdconfig));

    const std::vector<std::string> ve_in{"noisy_latent", "text_emb", "style_ttl",
                                         "latent_mask", "text_mask", "current_step", "total_step"};
    const std::vector<std::string> ve_out{"denoised"};
    g_ve.reset(Module::load(ve_in, ve_out, ve_mnn.c_str(), nullptr, &g_mdconfig));

    const std::vector<std::string> voc_in{"latent"};
    const std::vector<std::string> voc_out{"wav"};
    g_voc.reset(Module::load(voc_in, voc_out, vocoder_mnn.c_str(), nullptr, &g_mdconfig));

    if (!g_dp || !g_te || !g_ve || !g_voc) {
        STNPRINT("ERROR: one or more modules failed to load\n");
    } else {
        STNPRINT("all 4 modules loaded\n");
    }
}

void set_fixed_noise(const std::vector<float>& noise) {
    g_fixed_noise = noise;
}

// Build an int32 [1, T] input from the actual token vector.
static VARP make_ids_dyn(const std::vector<int>& ids) {
    const int len = static_cast<int>(ids.size());
    VARP v = _Input({1, len}, NCHW, halide_type_of<int>());
    if (len > 0) {
        ::memcpy(v->writeMap<int>(), ids.data(), len * sizeof(int));
    }
    return v;
}

std::vector<float> synth(const std::vector<int>& text_ids,
                         const std::vector<int>& pinyin_ids,
                         const std::vector<int>& tone_ids,
                         const std::vector<int>& prosody_ids,
                         const std::vector<float>& text_mask,
                         const std::vector<float>& style_ttl,
                         const std::vector<float>& style_dp,
                         int seed,
                         float speed) {
    if (!g_dp || !g_te || !g_ve || !g_voc) {
        STNPRINT("ERROR: synth called before models loaded\n");
        return {};
    }

    const size_t t = text_ids.size();
    if (t == 0 || t > MAX_TEXT || pinyin_ids.size() != t ||
        tone_ids.size() != t || prosody_ids.size() != t ||
        text_mask.size() != t) {
        STNPRINT("ERROR: inconsistent dynamic text inputs: text=%zu pinyin=%zu tone=%zu prosody=%zu mask=%zu\n",
                 t, pinyin_ids.size(), tone_ids.size(), prosody_ids.size(), text_mask.size());
        return {};
    }
    if (style_ttl.size() != STYLE_TTL_TOK * STYLE_TTL_DIM ||
        style_dp.size() != STYLE_DP_A * STYLE_DP_B) {
        STNPRINT("ERROR: invalid style sizes: ttl=%zu dp=%zu\n",
                 style_ttl.size(), style_dp.size());
        return {};
    }

    // ---- inputs shared across sub-models ----
    VARP v_text_ids = make_ids_dyn(text_ids);
    VARP v_pinyin_ids = make_ids_dyn(pinyin_ids);
    VARP v_tone_ids = make_ids_dyn(tone_ids);
    VARP v_prosody_ids = make_ids_dyn(prosody_ids);

    VARP v_text_mask = _Input({1, 1, static_cast<int>(t)}, NCHW, halide_type_of<float>());
    ::memcpy(v_text_mask->writeMap<float>(), text_mask.data(), t * sizeof(float));

    VARP v_style_ttl = _Input({1, STYLE_TTL_TOK, STYLE_TTL_DIM}, NCHW, halide_type_of<float>());
    ::memcpy(v_style_ttl->writeMap<float>(), style_ttl.data(), style_ttl.size() * sizeof(float));

    VARP v_style_dp = _Input({1, STYLE_DP_A, STYLE_DP_B}, NCHW, halide_type_of<float>());
    ::memcpy(v_style_dp->writeMap<float>(), style_dp.data(), style_dp.size() * sizeof(float));

    // ---- 1. DP -> duration (seconds) -> valid latent length ----
    auto dp_out = g_dp->onForward({v_text_ids, v_style_dp, v_text_mask});
    float dur_sec = dp_out[0]->readMap<float>()[0];
    // optional speed control: dur /= speed. Default 1.0 = model's native pace
    // (our v5 pipeline has NO speed factor — mnn_v5_eval.py uses raw duration).
    // >1.0 speeds up, <1.0 slows down. Kept as a user knob, off by default.
    if (speed > 0.0f) dur_sec /= speed;
    dur_sec = clampv(dur_sec, 2.5f, (float)(MAX_L_TTL * CHUNK_SAMPLES) / SR);
    // valid_l = floor(dur*SR / 3072). NOTE: our v5 pipeline (mnn_v5_eval.py) uses
    // int()-truncation everywhere, NOT ceil — the model was trained/validated
    // with this exact latent length. Official supertonic cpp uses ceil, but that
    // is for their EN model; we match OUR pipeline for parity with the model.
    int valid_l = clampv((int)(dur_sec * SR / (double)CHUNK_SAMPLES), 1, MAX_L_TTL);
    STNPERF("perf stage=tts_dp dur=%.2fs valid_l=%d (speed=%.2f)\n", dur_sec, valid_l, speed);

    // Dynamic latent shape: there is no padded tail to mask or crop.
    VARP v_latent_mask = _Input({1, 1, valid_l}, NCHW, halide_type_of<float>());
    float* latent_mask_ptr = v_latent_mask->writeMap<float>();
    for (int i = 0; i < valid_l; ++i) latent_mask_ptr[i] = 1.0f;

    // ---- 2. TE -> text_emb [1,256,T] ----
    auto te_out = g_te->onForward({v_text_ids, v_pinyin_ids, v_tone_ids,
                                   v_prosody_ids, v_style_ttl, v_text_mask});
    VARP text_emb = te_out[0];

    // ---- 3. VE 8-step Euler ----
    // initial latent x: [1,144,valid_l]
    std::vector<float> noise(LATENT_DIM * valid_l);
    if ((int)g_fixed_noise.size() >= LATENT_DIM * valid_l) {
        // Parity mode accepts channel-major full-bucket noise and takes each
        // channel's valid prefix (not one contiguous prefix of the flat array).
        for (int c = 0; c < LATENT_DIM; ++c) {
            std::copy(g_fixed_noise.begin() + c * MAX_L_TTL,
                      g_fixed_noise.begin() + c * MAX_L_TTL + valid_l,
                      noise.begin() + c * valid_l);
        }
    } else {
        std::mt19937 gen(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : noise) v = nd(gen);
    }
    VARP x = _Input({1, LATENT_DIM, valid_l}, NCHW, halide_type_of<float>());
    ::memcpy(x->writeMap<float>(), noise.data(), noise.size() * sizeof(float));

    VARP v_total = _Input({1}, NCHW, halide_type_of<float>());
    v_total->writeMap<float>()[0] = (float)TOTAL_STEP;

    for (int step = 0; step < TOTAL_STEP; ++step) {
        VARP v_cur = _Input({1}, NCHW, halide_type_of<float>());
        v_cur->writeMap<float>()[0] = (float)step;
        auto ve_out = g_ve->onForward({x, text_emb, v_style_ttl, v_latent_mask,
                                       v_text_mask, v_cur, v_total});
        // x = denoised * latent_mask
        x = ve_out[0] * v_latent_mask;
    }

    // ---- 4. Vocoder -> wav ----
    auto voc_out = g_voc->onForward({x});
    VARP wav = voc_out[0];
    auto info = wav->getInfo();
    int n = 1;
    for (int d : info->dim) n *= d;
    const float* wptr = wav->readMap<float>();

    int valid_samples = valid_l * CHUNK_SAMPLES;  // 512*6 = 3072 samples/frame
    if (valid_samples != n) {
        STNPRINT("WARNING: dynamic vocoder output=%d expected=%d samples\n", n, valid_samples);
    }
    std::vector<float> pcm(wptr, wptr + n);
    STNPERF("perf stage=tts_done samples=%d sec=%.2f dynamic valid_l=%d\n",
             n, (float)n / SR, valid_l);
    return pcm;
}

}  // namespace SUPERTONIC_NEKO

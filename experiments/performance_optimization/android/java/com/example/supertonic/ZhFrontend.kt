package com.example.supertonic

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest
import java.text.Normalizer

/**
 * Kotlin re-implementation of the training-time ZhTokenizer (parallel_pinyin
 * frontend) in supertonic/data/tokenizer.py.
 *
 * Produces the 4 id streams + mask that the TE consumes:
 *   text_ids     : unicode_indexer lookup of each char in "<zh>...</zh>"
 *   pinyin_ids   : MD5(base_pinyin) -> id, base from pinyin_dict.json
 *   tone_ids     : tone 1-5 from pinyin_dict.json
 *   prosody_ids  : punctuation/space rules
 *
 * Dynamic experiment: all streams contain exactly the processed token length.
 *
 * Parity is verified against Python: see deploy_android/scripts/gen_pinyin_dict.py
 * (validate() reports 0 mismatches) and the parity test in README.
 */
class ZhFrontend(context: Context) {

    companion object {
        const val MAX_TEXT = 256
        const val PINYIN_VOCAB_SIZE = 512

        // punctuation normalization (ZhTokenizer._punct_repl), applied for
        // non-"char" frontends (parallel_pinyin included).
        private val PUNCT_REPL = mapOf(
            '，' to ",", '、' to ",", '。' to ".", '！' to "!", '？' to "?",
            '；' to ";", '：' to ":", '…' to ".", '～' to "~",
            '「' to "\"", '」' to "\"", '『' to "\"", '』' to "\"",
            '（' to "(", '）' to ")", '【' to " ", '】' to " ",
        )

        // ZhTokenizer._char_repl
        private val CHAR_REPL = mapOf(
            '–' to "-", '‑' to "-", '—' to "-", '_' to " ",
            '“' to "\"", '”' to "\"", '‘' to "'", '’' to "'",
            '´' to "'", '`' to "'", '[' to " ", ']' to " ", '|' to " ", '/' to " ",
            '#' to " ", '→' to " ", '←' to " ",
        )
    }

    // unicode_indexer.json: a flat JSON array of length up to 65536, indexer[ord(c)] -> token id (or -1)
    private val indexer: IntArray
    // pinyin_dict.json: { "你": ["ni", 3], ... }
    private val pinyinBase = HashMap<Char, String>()
    private val pinyinTone = HashMap<Char, Int>()

    init {
        // load unicode_indexer.json from assets
        val idxJson = context.assets.open("unicode_indexer.json").bufferedReader().use { it.readText() }
        val arr = JSONArray(idxJson)
        indexer = IntArray(arr.length()) { arr.getInt(it) }

        // load pinyin_dict.json from assets
        val pyJson = context.assets.open("pinyin_dict.json").bufferedReader().use { it.readText() }
        val obj = JSONObject(pyJson)
        val keys = obj.keys()
        while (keys.hasNext()) {
            val k = keys.next()
            if (k.isEmpty()) continue
            val ch = k[0]
            val v = obj.getJSONArray(k)
            pinyinBase[ch] = v.getString(0)
            pinyinTone[ch] = v.getInt(1)
        }
    }

    private fun isCjk(c: Char): Boolean = c.code in 0x4E00..0x9FFF

    // ZhTokenizer._pinyin_id: 1 + (md5(base)[:4] little-endian % (VOCAB-1))
    private fun pinyinId(base: String): Int {
        if (base.isEmpty()) return 0
        val digest = MessageDigest.getInstance("MD5").digest(base.toByteArray(Charsets.UTF_8))
        // first 4 bytes, little-endian, unsigned
        val v = (digest[0].toLong() and 0xFF) or
                ((digest[1].toLong() and 0xFF) shl 8) or
                ((digest[2].toLong() and 0xFF) shl 16) or
                ((digest[3].toLong() and 0xFF) shl 24)
        return 1 + (v % (PINYIN_VOCAB_SIZE - 1)).toInt()
    }

    // ZhTokenizer._prosody_id
    private fun prosodyId(c: Char): Int = when (c) {
        ',', '，', '、' -> 1
        '.', '。' -> 2
        '?', '？' -> 3
        '!', '！' -> 4
        ';', '；', ':', '：' -> 5
        else -> if (c.isWhitespace()) 6 else 0
    }

    // ZhTokenizer._normalize_surface (parallel_pinyin path applies punct repl too)
    private fun normalizeSurface(text: String): String {
        var t = Normalizer.normalize(text, Normalizer.Form.NFKD)
        // emoji removal: iterate by CODE POINT (not char) so surrogate-pair emoji
        // like 😊 (U+1F60A, 2 UTF-16 chars) are matched. Matches _emoji_re intent.
        val sbEmoji = StringBuilder()
        var i = 0
        while (i < t.length) {
            val cp = t.codePointAt(i)
            val isEmoji = cp in 0x1F300..0x1FAFF || cp in 0x2600..0x27BF ||
                    cp in 0x1F1E6..0x1F1FF
            if (!isEmoji) sbEmoji.appendCodePoint(cp)
            i += Character.charCount(cp)
        }
        t = sbEmoji.toString()
        val sb = StringBuilder()
        for (c in t) sb.append(CHAR_REPL[c] ?: PUNCT_REPL[c] ?: c.toString())
        t = sb.toString()
        t = t.replace(Regex("[♥☆♡©\\\\]"), "")
        return t.replace(Regex("\\s+"), " ").trim()
    }

    // ZhTokenizer._preprocess (parallel_pinyin: no _apply_frontend transform, keeps chars)
    private fun preprocess(text: String): String {
        var t = normalizeSurface(text)
        if (!Regex(".*[.!?;:,'\"')\\]}…。」』】〉》›»]$").matches(t)) t += "."
        t = "<zh>$t</zh>"
        if (t.length > MAX_TEXT) {
            val keep = maxOf(1, MAX_TEXT - "</zh>".length)
            t = t.substring(0, keep).trimEnd() + "</zh>"
        }
        return t
    }

    /** Tokenized result: 4 id streams + mask, all exactly length n. */
    class Result(
        val textIds: IntArray,
        val pinyinIds: IntArray,
        val toneIds: IntArray,
        val prosodyIds: IntArray,
        val textMask: FloatArray,
    )

    fun tokenize(text: String): Result {
        val proc = preprocess(text)
        val n = minOf(proc.length, MAX_TEXT)

        val textIds = IntArray(n)
        val pinyinIds = IntArray(n)
        val toneIds = IntArray(n)
        val prosodyIds = IntArray(n)
        val mask = FloatArray(n)

        for (i in 0 until n) {
            val c = proc[i]
            // text_ids: indexer[ord(c)] if in range and >=0 else 0
            val cp = c.code
            textIds[i] = if (cp < indexer.size && indexer[cp] >= 0) indexer[cp] else 0
            mask[i] = 1.0f
            if (isCjk(c)) {
                val base = pinyinBase[c] ?: ""
                pinyinIds[i] = pinyinId(base)
                toneIds[i] = pinyinTone[c] ?: 0
                prosodyIds[i] = 0
            } else {
                pinyinIds[i] = 0
                toneIds[i] = 0
                prosodyIds[i] = prosodyId(c)
            }
        }
        return Result(textIds, pinyinIds, toneIds, prosodyIds, mask)
    }
}

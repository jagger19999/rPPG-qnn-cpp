package com.jagger.rppgbench.ui;

import org.json.JSONObject;
import java.util.Locale;

public final class HrStatusFormatter {
    public static final class CardView {
        public final String primary;
        public final String secondary;
        public CardView(String primary, String secondary) {
            this.primary = primary;
            this.secondary = secondary;
        }
    }

    private HrStatusFormatter() {}

    public static CardView traditional(JSONObject json) {
        if (json == null || !json.optBoolean("heart_rate_available", false)
                || !json.optBoolean("heart_rate_valid", false)) {
            String method = json == null ? "" : json.optString("traditional_method", "");
            return new CardView("--", method.isEmpty() ? "不可用" : method + " · 不可用");
        }
        int bpm = (int) Math.round(json.optDouble("bpm", Double.NaN));
        String method = json.optString("traditional_method", "rPPG");
        return new CardView(Integer.toString(bpm), method.toUpperCase(Locale.US) + " · 可信");
    }

    public static CardView deep(JSONObject json) {
        if (json == null || !json.optBoolean("deep_enabled", false)) {
            return new CardView("--", "未启用");
        }
        boolean available = json.optBoolean("deep_result_available", false);
        boolean waveformValid = json.optBoolean("deep_result_valid", false);
        boolean stabilityValid = json.optBoolean("deep_stability_valid", false);
        double displayBpm = json.optDouble("deep_display_bpm", Double.NaN);
        if (!available) {
            return new CardView("--", "等待结果");
        }
        boolean trusted = available && waveformValid && stabilityValid
                && Double.isFinite(displayBpm) && displayBpm > 0.0;

        String prefix;
        if (trusted) {
            prefix = "可信";
        } else if (!waveformValid) {
            prefix = nonEmpty(json.optString("deep_invalid_reason", ""), "信号无效");
        } else {
            prefix = nonEmpty(json.optString("deep_correction_reason", ""), "结果不稳定");
        }

        StringBuilder secondary = new StringBuilder(prefix);
        appendRoundedBpm(secondary, "原始", json.optDouble("deep_raw_bpm", Double.NaN));
        appendDecimal(secondary, "置信", json.optDouble("deep_confidence", Double.NaN), 2, "");
        appendDecimal(secondary, "窗口", json.optDouble(
                "deep_window_materialization_ms", Double.NaN), 0, "ms");
        appendDecimal(secondary, "预处理", json.optDouble(
                "deep_preprocess_ms", Double.NaN), 0, "ms");
        appendDecimal(secondary, "运行", json.optDouble("deep_runtime_ms", Double.NaN), 0, "ms");
        appendDecimal(secondary, "后处理", json.optDouble(
                "deep_postprocess_ms", Double.NaN), 0, "ms");
        appendDecimal(secondary, "总计", json.optDouble(
                "deep_inference_ms", Double.NaN), 0, "ms");

        return new CardView(
                trusted ? Integer.toString((int) Math.round(displayBpm)) : "--",
                secondary.toString());
    }

    private static String nonEmpty(String value, String fallback) {
        return value == null || value.isEmpty() ? fallback : value;
    }

    private static void appendRoundedBpm(StringBuilder builder, String label, double value) {
        if (Double.isFinite(value) && value > 0.0) {
            builder.append(" · ").append(label).append(' ').append(Math.round(value));
        }
    }

    private static void appendDecimal(
            StringBuilder builder, String label, double value, int decimals, String suffix) {
        if (!Double.isFinite(value) || value < 0.0) {
            return;
        }
        builder.append(" · ").append(label).append(' ')
                .append(String.format(Locale.US, "%." + decimals + "f", value))
                .append(suffix);
    }

    public static CardView watch(String status, Integer bpm, String errorCode) {
        String st = status == null ? "DISCONNECTED" : status;
        if (bpm == null) {
            String sec = errorCode != null ? st + " · " + errorCode : st;
            return new CardView("--", sec);
        }
        return new CardView(Integer.toString(bpm), st + " · 实验参考");
    }

    public static String alignmentLine(
            String status, Double absError, Double coverage) {
        if (status == null) return "对齐 --";
        if (absError == null || coverage == null) {
            return "对齐 " + status;
        }
        return String.format(Locale.US, "对齐 %s · |误差| %.1f · 覆盖 %.0f%%",
                status, absError, coverage * 100.0);
    }
}

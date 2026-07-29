package com.jagger.rppgbench.ui;

import org.json.JSONArray;
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

    public static String traditionalDiagnostic(JSONObject status) {
        if (status == null || !status.optBoolean("measurement_available", false)) {
            return "传统诊断: 等待首个 10 秒窗口";
        }
        JSONObject measurement = status.optJSONObject("measurement");
        if (measurement == null) {
            return "传统诊断: Snapshot 缺失";
        }

        long waveformRevision = status.optLong("traditional_waveform_revision", 0L);
        double measurementEnd = measurement.optDouble("window_end_sec", Double.NaN);
        double waveformEnd = status.optDouble(
                "traditional_waveform_window_end_sec", Double.NaN);
        boolean staleWaveform = waveformRevision > 0L
                && Double.isFinite(measurementEnd) && Double.isFinite(waveformEnd)
                && measurementEnd - waveformEnd > 0.5;
        String waveform = waveformRevision <= 0L ? "无"
                : (staleWaveform ? "历史" : "当前") + "(rev " + waveformRevision
                        + " end " + decimal(waveformEnd, 1) + "s)";
        boolean accepted = measurement.optBoolean("gate_accepted", false);
        String gateReason = measurement.optString("gate_reason", "unknown");
        String gate = accepted ? "接受" : "拒绝(" + gateReason + ")";
        String display = measurement.optBoolean("display_is_held", false)
                ? "保持旧值"
                : measurement.optBoolean("display_available", false) ? "新结果" : "无结果";

        StringBuilder output = new StringBuilder("传统诊断: 波形=")
                .append(waveform).append(" · Gate=").append(gate)
                .append(" · 显示=").append(display);
        JSONObject quality = measurement.optJSONObject("quality");
        if (quality != null) {
            output.append('\n').append("质量: ROI FPS=")
                    .append(decimal(quality.optDouble("source_fps", Double.NaN), 1))
                    .append(" · gap=")
                    .append(decimal(quality.optDouble("max_frame_gap_sec", Double.NaN), 3))
                    .append("s · face=").append(quality.optInt("face_count", 0))
                    .append(" · area=")
                    .append(decimal(quality.optDouble("face_area_ratio", Double.NaN), 3))
                    .append(" · motion=")
                    .append(decimal(quality.optDouble("motion_px", Double.NaN), 1))
                    .append("px · brightness=")
                    .append(decimal(quality.optDouble("brightness", Double.NaN), 3))
                    .append(" · signal_std=")
                    .append(decimal(quality.optDouble("signal_std", Double.NaN), 4))
                    .append(" · peak=")
                    .append(decimal(quality.optDouble("spectral_peak_ratio", Double.NaN), 3));
        }
        JSONArray candidates = measurement.optJSONArray("candidates");
        if (candidates != null && candidates.length() > 0) {
            output.append('\n').append("候选: ");
            for (int index = 0; index < candidates.length(); index++) {
                if (index > 0) output.append(" · ");
                JSONObject candidate = candidates.optJSONObject(index);
                if (candidate == null) continue;
                output.append(candidate.optString("method", "?"));
                if (candidate.optBoolean("valid", false)) {
                    output.append('=').append(decimal(
                            candidate.optDouble("bpm", Double.NaN), 1))
                            .append(" bpm(conf ")
                            .append(decimal(candidate.optDouble(
                                    "confidence", Double.NaN), 2)).append(')');
                } else {
                    output.append("=无效(")
                            .append(nonEmpty(candidate.optString(
                                    "invalid_reason", ""), "unknown"))
                            .append(')');
                }
            }
        }
        return output.toString();
    }

    private static String decimal(double value, int decimals) {
        return Double.isFinite(value)
                ? String.format(Locale.US, "%." + decimals + "f", value) : "--";
    }

    public static CardView traditional(JSONObject json) {
        if (json != null && json.optBoolean("measurement_available", false)) {
            JSONObject measurement = json.optJSONObject("measurement");
            if (measurement != null && measurement.optBoolean("display_available", false)) {
                int bpm = (int) Math.round(measurement.optDouble("display_bpm", Double.NaN));
                if (measurement.optBoolean("display_is_held", false)) {
                    String reason = measurement.optString("gate_reason", "质量不足");
                    return new CardView(Integer.toString(bpm), "保持上一可信值 · " + reason);
                }
                String method = measurement.optString("selected_method", "CONSENSUS");
                String vqa = measurement.optString("vqa_label", "");
                String suffix = vqa.isEmpty() ? "可信" : vqa + " · 可信";
                return new CardView(Integer.toString(bpm), method + " · " + suffix);
            }
            String reason = measurement == null ? "采样中"
                    : measurement.optString("gate_reason", "采样中");
            return new CardView("--", "不可用 · " + reason);
        }
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

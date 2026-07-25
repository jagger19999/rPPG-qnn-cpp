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
        if (!json.optBoolean("deep_result_available", false)
                || !json.optBoolean("deep_result_valid", false)) {
            String reason = json.optString("deep_invalid_reason", "不可用");
            if (reason.isEmpty()) reason = "不可用";
            return new CardView("--", reason);
        }
        int bpm = (int) Math.round(json.optDouble("deep_bpm", Double.NaN));
        double ms = json.optDouble("deep_inference_ms", Double.NaN);
        String secondary = "ORT CPU";
        if (Double.isFinite(ms)) {
            secondary = String.format(Locale.US, "ORT CPU · %.0fms", ms);
        }
        return new CardView(Integer.toString(bpm), secondary);
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

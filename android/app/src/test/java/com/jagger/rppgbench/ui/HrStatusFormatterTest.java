package com.jagger.rppgbench.ui;

import org.json.JSONObject;
import org.junit.Test;
import static org.junit.Assert.*;

public class HrStatusFormatterTest {
    @Test
    public void traditionalDiagnosticExplainsWaveformWithoutAcceptedBpm() throws Exception {
        JSONObject json = new JSONObject(
                "{\"measurement_available\":true,\"traditional_waveform_revision\":8,"
                        + "\"traditional_waveform_window_end_sec\":30.0,"
                        + "\"measurement\":{\"window_end_sec\":31.5,"
                        + "\"gate_accepted\":false,\"gate_reason\":\"low_fps\","
                        + "\"display_available\":false,\"display_is_held\":false,"
                        + "\"quality\":{\"source_fps\":12.4,\"max_frame_gap_sec\":0.18,"
                        + "\"brightness\":0.48,\"signal_std\":0.012,"
                        + "\"spectral_peak_ratio\":0.22,\"face_area_ratio\":0.16,"
                        + "\"motion_px\":3.2,\"face_count\":1},"
                        + "\"candidates\":[{\"method\":\"GREEN\",\"valid\":false,"
                        + "\"invalid_reason\":\"low_source_fps\",\"bpm\":0,"
                        + "\"confidence\":0,\"peak_ratio\":0}]}}");

        String diagnostic = HrStatusFormatter.traditionalDiagnostic(json);

        assertTrue(diagnostic.contains("波形=历史(rev 8"));
        assertTrue(diagnostic.contains("Gate=拒绝(low_fps)"));
        assertTrue(diagnostic.contains("ROI FPS=12.4"));
        assertTrue(diagnostic.contains("GREEN=无效(low_source_fps)"));
    }

    @Test
    public void traditionalDiagnosticExplainsWaitingForFirstWindow() throws Exception {
        JSONObject json = new JSONObject(
                "{\"measurement_available\":false,\"traditional_waveform_revision\":0}");

        assertTrue(HrStatusFormatter.traditionalDiagnostic(json).contains("等待首个 10 秒窗口"));
    }

    @Test
    public void measurementSnapshotDistinguishesAcceptedAndHeldValues() throws Exception {
        JSONObject accepted = new JSONObject(
                "{\"measurement_available\":true,\"measurement\":{" +
                "\"display_available\":true,\"display_bpm\":72.0," +
                "\"display_is_held\":false,\"selected_method\":\"CONSENSUS\"," +
                "\"vqa_label\":\"Excellent\"}}");
        HrStatusFormatter.CardView acceptedView = HrStatusFormatter.traditional(accepted);
        assertEquals("72", acceptedView.primary);
        assertTrue(acceptedView.secondary.contains("可信"));

        JSONObject held = new JSONObject(
                "{\"measurement_available\":true,\"measurement\":{" +
                "\"display_available\":true,\"display_bpm\":72.0," +
                "\"display_is_held\":true,\"gate_reason\":\"high_motion\"}}");
        HrStatusFormatter.CardView heldView = HrStatusFormatter.traditional(held);
        assertEquals("72", heldView.primary);
        assertTrue(heldView.secondary.contains("保持"));
    }
    @Test
    public void traditionalShowsDashWhenInvalid() throws Exception {
        JSONObject json = new JSONObject(
                "{\"heart_rate_available\":true,\"heart_rate_valid\":false,"
                        + "\"bpm\":0,\"traditional_method\":\"green\"}");
        HrStatusFormatter.CardView traditional = HrStatusFormatter.traditional(json);
        assertEquals("--", traditional.primary);
        assertTrue(traditional.secondary.contains("不可用"));
    }

    @Test
    public void traditionalShowsBpmWhenValid() throws Exception {
        JSONObject json = new JSONObject(
                "{\"heart_rate_available\":true,\"heart_rate_valid\":true,"
                        + "\"bpm\":72.4,\"traditional_method\":\"pos\"}");
        HrStatusFormatter.CardView traditional = HrStatusFormatter.traditional(json);
        assertEquals("72", traditional.primary);
        assertTrue(traditional.secondary.toLowerCase().contains("pos"));
    }

    @Test
    public void deepShowsDashWhenDisabled() throws Exception {
        JSONObject json = new JSONObject(
                "{\"deep_enabled\":false,\"deep_result_available\":false}");
        assertEquals("--", HrStatusFormatter.deep(json).primary);
    }

    @Test
    public void deepShowsStabilizedAndRawBpmWithAllTimingStages() throws Exception {
        JSONObject json = new JSONObject(
                "{\"deep_enabled\":true,\"deep_result_available\":true,"
                        + "\"deep_result_valid\":true,\"deep_stability_valid\":true,"
                        + "\"deep_display_bpm\":71.6,\"deep_raw_bpm\":143.2,"
                        + "\"deep_confidence\":0.82,"
                        + "\"deep_window_materialization_ms\":18.2,"
                        + "\"deep_preprocess_ms\":41.1,\"deep_runtime_ms\":912.4,"
                        + "\"deep_postprocess_ms\":7.3,\"deep_inference_ms\":960.8}");

        HrStatusFormatter.CardView card = HrStatusFormatter.deep(json);

        assertEquals("72", card.primary);
        assertEquals(
                "可信 · 原始 143 · 置信 0.82 · 窗口 18ms · 预处理 41ms · "
                        + "运行 912ms · 后处理 7ms · 总计 961ms",
                card.secondary);
    }

    @Test
    public void deepNeverPresentsUnstableOrLowConfidenceResultAsTrusted() throws Exception {
        JSONObject json = new JSONObject(
                "{\"deep_enabled\":true,\"deep_result_available\":true,"
                        + "\"deep_result_valid\":true,\"deep_stability_valid\":false,"
                        + "\"deep_display_bpm\":72,\"deep_raw_bpm\":144,"
                        + "\"deep_confidence\":0.12,"
                        + "\"deep_correction_reason\":\"low_confidence\","
                        + "\"deep_inference_ms\":900}");

        HrStatusFormatter.CardView card = HrStatusFormatter.deep(json);

        assertEquals("--", card.primary);
        assertEquals("low_confidence · 原始 144 · 置信 0.12 · 总计 900ms", card.secondary);
    }

    @Test
    public void watchShowsBpmAndStatus() {
        HrStatusFormatter.CardView watch =
                HrStatusFormatter.watch("STREAMING", 70, null);
        assertEquals("70", watch.primary);
        assertTrue(watch.secondary.contains("STREAMING"));
    }
}

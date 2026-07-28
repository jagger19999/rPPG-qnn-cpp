package com.jagger.rppgbench.ui;

import org.json.JSONObject;
import org.junit.Test;
import static org.junit.Assert.*;

public class HrStatusFormatterTest {
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
    public void watchShowsBpmAndStatus() {
        HrStatusFormatter.CardView watch =
                HrStatusFormatter.watch("STREAMING", 70, null);
        assertEquals("70", watch.primary);
        assertTrue(watch.secondary.contains("STREAMING"));
    }
}

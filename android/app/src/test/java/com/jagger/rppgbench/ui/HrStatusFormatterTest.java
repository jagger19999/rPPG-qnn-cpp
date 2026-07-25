package com.jagger.rppgbench.ui;

import org.json.JSONObject;
import org.junit.Test;
import static org.junit.Assert.*;

public class HrStatusFormatterTest {
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

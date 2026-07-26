package com.jagger.rppgbench.ui;

import java.util.Locale;

public final class PpgWaveformState {
    private PpgWaveformState() {}

    public static String deep(String modelLabel, boolean enabled, int collected, int required,
            boolean resultAvailable, boolean valid, String reason) {
        if (!enabled) {
            return "未启用深度推理";
        }
        if (!resultAvailable && collected < required) {
            return "正在采集 " + modelLabel + " 窗口 " + collected + " / " + required;
        }
        if (!resultAvailable) {
            return modelLabel + " 推理中";
        }
        if (!valid) {
            String detail = reason == null || reason.isEmpty() ? "不可用" : reason;
            return "信号无效：" + detail;
        }
        return String.format(Locale.US, "%s · %d 点 · -%.1f s → 0 s",
                modelLabel, required, required / 30.0);
    }
}

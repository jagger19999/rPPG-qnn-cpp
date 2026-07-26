package com.jagger.rppgbench.ui;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class PpgWaveformStateTest {
    @Test
    public void deepStatesMatchProductRules() {
        assertEquals("未启用深度推理", PpgWaveformState.deep("深度模型", false, 0, 180, false, false, ""));
        assertEquals("正在采集 EfficientPhys 窗口 73 / 180", PpgWaveformState.deep("EfficientPhys", true, 73, 180, false, false, ""));
        assertEquals("EfficientPhys 推理中", PpgWaveformState.deep("EfficientPhys", true, 180, 180, false, false, ""));
        assertEquals("信号无效：low_confidence", PpgWaveformState.deep("EfficientPhys", true, 180, 180, true, false, "low_confidence"));
        assertEquals("EfficientPhys · 180 点 · -6.0 s → 0 s", PpgWaveformState.deep("EfficientPhys", true, 180, 180, true, true, ""));
    }
}

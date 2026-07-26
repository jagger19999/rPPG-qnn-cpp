package com.jagger.rppgbench.ui;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class PpgWaveformStateTest {
    @Test
    public void deepStatesMatchProductRules() {
        assertEquals("未启用深度推理", PpgWaveformState.deep(false, 0, 180, false, false, ""));
        assertEquals("正在采集深度窗口 73 / 180", PpgWaveformState.deep(true, 73, 180, false, false, ""));
        assertEquals("TSCAN 推理中", PpgWaveformState.deep(true, 180, 180, false, false, ""));
        assertEquals("信号无效：waveform_invalid", PpgWaveformState.deep(true, 180, 180, true, false, "waveform_invalid"));
        assertEquals("TSCAN · 180 帧 · 约 6.0 秒", PpgWaveformState.deep(true, 180, 180, true, true, ""));
    }
}

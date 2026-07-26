package com.jagger.rppgbench.ui;

import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

public final class PpgWaveformGeometryTest {
    @Test
    public void mapsSignalToChartBounds() {
        float[] points = new float[6];
        PpgWaveformGeometry.fillPoints(new float[] {-1f, 0f, 1f}, 10f, 20f, 100f, 40f, points);
        assertArrayEquals(new float[] {10f, 60f, 60f, 40f, 110f, 20f}, points, 0.001f);
    }
}

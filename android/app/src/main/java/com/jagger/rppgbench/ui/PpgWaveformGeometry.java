package com.jagger.rppgbench.ui;

public final class PpgWaveformGeometry {
    private PpgWaveformGeometry() {}

    public static void fillPoints(float[] values, float left, float top,
            float width, float height, float[] points) {
        if (values == null || values.length < 2 || points == null
                || points.length < values.length * 2) {
            throw new IllegalArgumentException("insufficient waveform geometry data");
        }
        float step = width / (values.length - 1);
        float middle = top + height / 2f;
        for (int index = 0; index < values.length; index++) {
            points[index * 2] = left + index * step;
            points[index * 2 + 1] = middle - values[index] * height / 2f;
        }
    }
}

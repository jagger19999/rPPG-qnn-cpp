package com.jagger.rppgbench.ui;

public final class PpgWaveformSnapshot {
    public final long revision;
    public final String method;
    public final double sampleRateHz;
    public final boolean valid;
    public final String invalidReason;
    public final double windowEndSec;
    private final float[] values;

    public PpgWaveformSnapshot(long revision, String method, double sampleRateHz,
            boolean valid, String invalidReason, double windowEndSec, float[] values) {
        if (revision < 0 || !Double.isFinite(sampleRateHz) || sampleRateHz <= 0
                || !Double.isFinite(windowEndSec) || values == null || values.length < 2) {
            throw new IllegalArgumentException("invalid waveform metadata");
        }
        for (float value : values) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException("waveform must be finite");
            }
        }
        this.revision = revision;
        this.method = method == null ? "" : method;
        this.sampleRateHz = sampleRateHz;
        this.valid = valid;
        this.invalidReason = invalidReason == null ? "" : invalidReason;
        this.windowEndSec = windowEndSec;
        this.values = values.clone();
    }

    public float[] values() {
        return values.clone();
    }

    public int sampleCount() {
        return values.length;
    }

    public double relativeStartSeconds() {
        return -(values.length - 1) / sampleRateHz;
    }
}

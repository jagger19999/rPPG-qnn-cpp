package com.jagger.rppgbench.ui;

import static org.junit.Assert.*;

import org.junit.Test;

public final class PpgWaveformSnapshotTest {
    @Test
    public void snapshotDefensivelyCopiesAndCalculatesDuration() {
        float[] input = {-1f, 0f, 1f};
        PpgWaveformSnapshot snapshot = new PpgWaveformSnapshot(
                4, "POS", 2.0, true, "", 12.5, input);
        input[0] = 9f;
        assertArrayEquals(new float[] {-1f, 0f, 1f}, snapshot.values(), 0f);
        assertEquals(-1.0, snapshot.relativeStartSeconds(), 1e-9);
        assertEquals(12.5, snapshot.windowEndSec, 1e-9);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectsNonFiniteValues() {
        new PpgWaveformSnapshot(
                1, "TSCAN", 30.0, false, "bad", 10.0,
                new float[] {0f, Float.NaN});
    }
}

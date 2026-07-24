package com.jagger.rppgbench.watch;

import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

public final class WatchAlignerTest {
    @Test
    public void disconnectedSnapshot() {
        WatchContracts.WatchHeartRateSnapshot snapshot =
                new WatchContracts.WatchHeartRateSnapshot(
                        WatchContracts.WatchConnectionStatus.DISCONNECTED,
                        Collections.emptyList(),
                        Collections.emptyList(),
                        null,
                        0,
                        0);
        WatchContracts.WatchAlignmentResult result =
                WatchAligner.align(new WatchAligner.RppgWindow(0, 10, 70.0, true), snapshot, 100.0);
        assertEquals(WatchContracts.WatchAlignmentStatus.DISCONNECTED, result.status);
        assertNull(result.watchReferenceBpm);
    }

    @Test
    public void staleSnapshot() {
        WatchContracts.WatchHeartRateSnapshot snapshot =
                new WatchContracts.WatchHeartRateSnapshot(
                        WatchContracts.WatchConnectionStatus.STALE,
                        Collections.singletonList(
                                new WatchContracts.WatchHeartRateSample(101.0, 72, new double[0], "a", "w")),
                        Collections.emptyList(),
                        null,
                        0,
                        0);
        WatchContracts.WatchAlignmentResult result =
                WatchAligner.align(new WatchAligner.RppgWindow(0, 10, 70.0, true), snapshot, 100.0);
        assertEquals(WatchContracts.WatchAlignmentStatus.WATCH_STALE, result.status);
    }

    @Test
    public void invalidRppg() {
        WatchContracts.WatchHeartRateSnapshot snapshot = streamingSamples();
        WatchContracts.WatchAlignmentResult result =
                WatchAligner.align(new WatchAligner.RppgWindow(0, 10, null, false), snapshot, 100.0);
        assertEquals(WatchContracts.WatchAlignmentStatus.RPPG_INVALID, result.status);
    }

    @Test
    public void partialCoverageWhenTooFewSamples() {
        WatchContracts.WatchHeartRateSnapshot snapshot =
                new WatchContracts.WatchHeartRateSnapshot(
                        WatchContracts.WatchConnectionStatus.STREAMING,
                        Arrays.asList(
                                new WatchContracts.WatchHeartRateSample(101.0, 70, new double[0], "a", "w"),
                                new WatchContracts.WatchHeartRateSample(102.0, 71, new double[0], "a", "w")),
                        Collections.emptyList(),
                        null,
                        0,
                        0);
        WatchContracts.WatchAlignmentResult result =
                WatchAligner.align(new WatchAligner.RppgWindow(0, 10, 72.0, true), snapshot, 100.0);
        assertEquals(WatchContracts.WatchAlignmentStatus.PARTIAL_COVERAGE, result.status);
        assertEquals(2, result.watchSampleCount);
    }

    @Test
    public void alignedUsesMedianAndErrors() {
        WatchContracts.WatchHeartRateSnapshot snapshot = streamingSamples();
        WatchContracts.WatchAlignmentResult result =
                WatchAligner.align(new WatchAligner.RppgWindow(0, 10, 74.0, true), snapshot, 100.0);
        assertEquals(WatchContracts.WatchAlignmentStatus.ALIGNED, result.status);
        assertEquals(71.0, result.watchReferenceBpm, 1e-9);
        assertEquals(3.0, result.signedErrorBpm, 1e-9);
        assertEquals(3.0, result.absoluteErrorBpm, 1e-9);
        assertEquals(6, result.watchSampleCount);
    }

    @Test
    public void summarizeAlignedOnly() {
        WatchContracts.WatchAlignmentResult aligned =
                new WatchContracts.WatchAlignmentResult(
                        0, 10, 74.0, 71.0, 3.0, 3.0, 4, 0.9, 1.0,
                        WatchContracts.WatchAlignmentStatus.ALIGNED);
        WatchContracts.WatchAlignmentResult partial =
                new WatchContracts.WatchAlignmentResult(
                        1, 11, 74.0, null, null, null, 1, 0.1, null,
                        WatchContracts.WatchAlignmentStatus.PARTIAL_COVERAGE);
        WatchAligner.Summary summary = WatchAligner.summarize(Arrays.asList(aligned, partial));
        assertEquals(2, summary.totalWindows);
        assertEquals(1, summary.alignedWindows);
        assertEquals(0.5, summary.validWindowRatio, 1e-9);
        assertEquals(3.0, summary.maeBpm, 1e-9);
    }

    private static WatchContracts.WatchHeartRateSnapshot streamingSamples() {
        return new WatchContracts.WatchHeartRateSnapshot(
                WatchContracts.WatchConnectionStatus.STREAMING,
                Arrays.asList(
                        new WatchContracts.WatchHeartRateSample(100.5, 70, new double[0], "a", "w"),
                        new WatchContracts.WatchHeartRateSample(102.0, 71, new double[0], "a", "w"),
                        new WatchContracts.WatchHeartRateSample(104.0, 72, new double[0], "a", "w"),
                        new WatchContracts.WatchHeartRateSample(106.0, 71, new double[0], "a", "w"),
                        new WatchContracts.WatchHeartRateSample(108.0, 71, new double[0], "a", "w"),
                        new WatchContracts.WatchHeartRateSample(109.5, 72, new double[0], "a", "w")),
                Collections.emptyList(),
                null,
                0,
                0);
    }
}

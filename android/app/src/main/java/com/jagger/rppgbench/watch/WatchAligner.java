package com.jagger.rppgbench.watch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class WatchAligner {
    public static final class RppgWindow {
        public final double startSec;
        public final double endSec;
        public final Double bpm;
        public final boolean valid;

        public RppgWindow(double startSec, double endSec, Double bpm, boolean valid) {
            this.startSec = startSec;
            this.endSec = endSec;
            this.bpm = bpm;
            this.valid = valid;
        }
    }

    public static final class Summary {
        public final int totalWindows;
        public final int alignedWindows;
        public final double validWindowRatio;
        public final Double maeBpm;
        public final Double rmseBpm;
        public final Double meanBiasBpm;

        public Summary(
                int totalWindows,
                int alignedWindows,
                double validWindowRatio,
                Double maeBpm,
                Double rmseBpm,
                Double meanBiasBpm) {
            this.totalWindows = totalWindows;
            this.alignedWindows = alignedWindows;
            this.validWindowRatio = validWindowRatio;
            this.maeBpm = maeBpm;
            this.rmseBpm = rmseBpm;
            this.meanBiasBpm = meanBiasBpm;
        }
    }

    private WatchAligner() {
    }

    public static WatchContracts.WatchAlignmentResult align(
            RppgWindow window,
            WatchContracts.WatchHeartRateSnapshot snapshot,
            double sessionStartMonotonicSec) {
        if (snapshot.status == WatchContracts.WatchConnectionStatus.DISCONNECTED
                || snapshot.status == WatchContracts.WatchConnectionStatus.ERROR) {
            return empty(window, WatchContracts.WatchAlignmentStatus.DISCONNECTED);
        }
        if (snapshot.status == WatchContracts.WatchConnectionStatus.STALE) {
            return empty(window, WatchContracts.WatchAlignmentStatus.WATCH_STALE);
        }
        if (!window.valid || window.bpm == null) {
            return empty(window, WatchContracts.WatchAlignmentStatus.RPPG_INVALID);
        }

        List<WatchContracts.WatchHeartRateSample> inWindow = new ArrayList<>();
        for (WatchContracts.WatchHeartRateSample sample : snapshot.samples) {
            double relative = sample.receivedMonotonicSec - sessionStartMonotonicSec;
            if (relative >= window.startSec && relative <= window.endSec) {
                inWindow.add(sample);
            }
        }
        inWindow.sort((left, right) ->
                Double.compare(left.receivedMonotonicSec, right.receivedMonotonicSec));

        List<Double> times = new ArrayList<>();
        for (WatchContracts.WatchHeartRateSample sample : inWindow) {
            times.add(sample.receivedMonotonicSec - sessionStartMonotonicSec);
        }
        double duration = Math.max(window.endSec - window.startSec, 1e-12);
        double coverage = times.size() < 2
                ? 0.0
                : Math.min(1.0, Math.max(0.0, (times.get(times.size() - 1) - times.get(0)) / duration));
        Double maxGap = null;
        if (times.size() >= 2) {
            double gap = 0.0;
            for (int index = 1; index < times.size(); index++) {
                gap = Math.max(gap, times.get(index) - times.get(index - 1));
            }
            maxGap = gap;
        }
        if (inWindow.size() < 3 || coverage < 0.70 || maxGap == null || maxGap > 2.0) {
            return new WatchContracts.WatchAlignmentResult(
                    window.startSec,
                    window.endSec,
                    window.bpm,
                    null,
                    null,
                    null,
                    inWindow.size(),
                    coverage,
                    maxGap,
                    WatchContracts.WatchAlignmentStatus.PARTIAL_COVERAGE);
        }

        List<Integer> bpms = new ArrayList<>();
        for (WatchContracts.WatchHeartRateSample sample : inWindow) {
            bpms.add(sample.bpm);
        }
        Collections.sort(bpms);
        double reference;
        int mid = bpms.size() / 2;
        if ((bpms.size() & 1) == 1) {
            reference = bpms.get(mid);
        } else {
            reference = 0.5 * (bpms.get(mid - 1) + bpms.get(mid));
        }
        double signed = window.bpm - reference;
        return new WatchContracts.WatchAlignmentResult(
                window.startSec,
                window.endSec,
                window.bpm,
                reference,
                signed,
                Math.abs(signed),
                inWindow.size(),
                coverage,
                maxGap,
                WatchContracts.WatchAlignmentStatus.ALIGNED);
    }

    public static Summary summarize(List<WatchContracts.WatchAlignmentResult> alignments) {
        List<WatchContracts.WatchAlignmentResult> aligned = new ArrayList<>();
        for (WatchContracts.WatchAlignmentResult item : alignments) {
            if (item.status == WatchContracts.WatchAlignmentStatus.ALIGNED) {
                aligned.add(item);
            }
        }
        int total = alignments.size();
        if (aligned.isEmpty()) {
            return new Summary(total, 0, total == 0 ? 0.0 : 0.0, null, null, null);
        }
        double absSum = 0.0;
        double sqSum = 0.0;
        double signedSum = 0.0;
        for (WatchContracts.WatchAlignmentResult item : aligned) {
            absSum += item.absoluteErrorBpm;
            sqSum += item.absoluteErrorBpm * item.absoluteErrorBpm;
            signedSum += item.signedErrorBpm;
        }
        int n = aligned.size();
        return new Summary(
                total,
                n,
                (double) n / (double) total,
                absSum / n,
                Math.sqrt(sqSum / n),
                signedSum / n);
    }

    private static WatchContracts.WatchAlignmentResult empty(
            RppgWindow window, WatchContracts.WatchAlignmentStatus status) {
        return new WatchContracts.WatchAlignmentResult(
                window.startSec,
                window.endSec,
                window.bpm,
                null,
                null,
                null,
                0,
                0.0,
                null,
                status);
    }
}

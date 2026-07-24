package com.jagger.rppgbench.watch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

public final class WatchContracts {
    public enum WatchConnectionStatus {
        DISCONNECTED,
        SCANNING,
        CONNECTING,
        STREAMING,
        STALE,
        RECONNECTING,
        ERROR
    }

    public enum WatchAlignmentStatus {
        PENDING,
        ALIGNED,
        PARTIAL_COVERAGE,
        WATCH_STALE,
        RPPG_INVALID,
        DISCONNECTED
    }

    public static final class WatchDevice {
        public final String id;
        public final String name;

        public WatchDevice(String id, String name) {
            this.id = Objects.requireNonNull(id, "id");
            this.name = Objects.requireNonNull(name, "name");
        }
    }

    public static final class WatchHeartRateSample {
        public final double receivedMonotonicSec;
        public final int bpm;
        public final double[] rrIntervalsSec;
        public final String deviceId;
        public final String deviceName;

        public WatchHeartRateSample(
                double receivedMonotonicSec,
                int bpm,
                double[] rrIntervalsSec,
                String deviceId,
                String deviceName) {
            if (bpm < 1 || bpm > 300) {
                throw new IllegalArgumentException("watch bpm must be in [1, 300]");
            }
            if (receivedMonotonicSec < 0.0) {
                throw new IllegalArgumentException("receivedMonotonicSec must be non-negative");
            }
            this.receivedMonotonicSec = receivedMonotonicSec;
            this.bpm = bpm;
            this.rrIntervalsSec = rrIntervalsSec == null ? new double[0] : rrIntervalsSec.clone();
            for (double value : this.rrIntervalsSec) {
                if (value <= 0.0) {
                    throw new IllegalArgumentException("RR intervals must be positive");
                }
            }
            this.deviceId = deviceId == null ? "" : deviceId;
            this.deviceName = deviceName == null ? "" : deviceName;
        }
    }

    public static final class WatchHeartRateSnapshot {
        public final WatchConnectionStatus status;
        public final WatchHeartRateSample latestSample;
        public final List<WatchHeartRateSample> samples;
        public final List<WatchDevice> devices;
        public final String errorCode;
        public final int malformedPackets;
        public final int reconnectAttempts;

        public WatchHeartRateSnapshot(
                WatchConnectionStatus status,
                List<WatchHeartRateSample> samples,
                List<WatchDevice> devices,
                String errorCode,
                int malformedPackets,
                int reconnectAttempts) {
            this.status = Objects.requireNonNull(status, "status");
            this.samples = Collections.unmodifiableList(new ArrayList<>(samples));
            this.devices = Collections.unmodifiableList(new ArrayList<>(devices));
            this.latestSample = this.samples.isEmpty() ? null : this.samples.get(this.samples.size() - 1);
            this.errorCode = errorCode;
            this.malformedPackets = malformedPackets;
            this.reconnectAttempts = reconnectAttempts;
        }

        public boolean isStale(double nowMonotonicSec, double staleAfterSec) {
            return latestSample == null
                    || nowMonotonicSec - latestSample.receivedMonotonicSec > staleAfterSec;
        }
    }

    public static final class WatchAlignmentResult {
        public final double startSec;
        public final double endSec;
        public final Double rppgBpm;
        public final Double watchReferenceBpm;
        public final Double signedErrorBpm;
        public final Double absoluteErrorBpm;
        public final int watchSampleCount;
        public final double coverageRatio;
        public final Double maxGapSec;
        public final WatchAlignmentStatus status;

        public WatchAlignmentResult(
                double startSec,
                double endSec,
                Double rppgBpm,
                Double watchReferenceBpm,
                Double signedErrorBpm,
                Double absoluteErrorBpm,
                int watchSampleCount,
                double coverageRatio,
                Double maxGapSec,
                WatchAlignmentStatus status) {
            this.startSec = startSec;
            this.endSec = endSec;
            this.rppgBpm = rppgBpm;
            this.watchReferenceBpm = watchReferenceBpm;
            this.signedErrorBpm = signedErrorBpm;
            this.absoluteErrorBpm = absoluteErrorBpm;
            this.watchSampleCount = watchSampleCount;
            this.coverageRatio = coverageRatio;
            this.maxGapSec = maxGapSec;
            this.status = Objects.requireNonNull(status, "status");
        }
    }

    private WatchContracts() {
    }
}

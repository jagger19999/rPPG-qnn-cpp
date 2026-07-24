package com.jagger.rppgbench.watch;

import java.util.ArrayList;
import java.util.List;

public final class WatchSampleStore {
    private static final double HISTORY_SEC = 180.0;
    private static final double STALE_AFTER_SEC = 2.0;

    private final Object lock = new Object();
    private WatchContracts.WatchConnectionStatus status =
            WatchContracts.WatchConnectionStatus.DISCONNECTED;
    private final List<WatchContracts.WatchHeartRateSample> samples = new ArrayList<>();
    private final List<WatchContracts.WatchDevice> devices = new ArrayList<>();
    private String errorCode;
    private int malformedPackets;
    private int reconnectAttempts;

    public void setStatus(WatchContracts.WatchConnectionStatus status) {
        synchronized (lock) {
            this.status = status;
            if (status != WatchContracts.WatchConnectionStatus.ERROR) {
                this.errorCode = null;
            }
        }
    }

    public void setDevices(List<WatchContracts.WatchDevice> devices) {
        synchronized (lock) {
            this.devices.clear();
            this.devices.addAll(devices);
        }
    }

    public void setError(String errorCode) {
        synchronized (lock) {
            this.errorCode = errorCode;
            this.status = WatchContracts.WatchConnectionStatus.ERROR;
        }
    }

    public void setReconnectAttempts(int reconnectAttempts) {
        synchronized (lock) {
            this.reconnectAttempts = reconnectAttempts;
        }
    }

    public void incrementMalformed() {
        synchronized (lock) {
            ++malformedPackets;
        }
    }

    public void acceptMeasurement(
            double receivedMonotonicSec,
            int bpm,
            double[] rrIntervalsSec,
            String deviceId,
            String deviceName) {
        WatchContracts.WatchHeartRateSample sample =
                new WatchContracts.WatchHeartRateSample(
                        receivedMonotonicSec, bpm, rrIntervalsSec, deviceId, deviceName);
        synchronized (lock) {
            samples.add(sample);
            pruneLocked(receivedMonotonicSec);
            if (status == WatchContracts.WatchConnectionStatus.CONNECTING
                    || status == WatchContracts.WatchConnectionStatus.RECONNECTING
                    || status == WatchContracts.WatchConnectionStatus.STALE
                    || status == WatchContracts.WatchConnectionStatus.STREAMING) {
                status = WatchContracts.WatchConnectionStatus.STREAMING;
                errorCode = null;
            }
        }
    }

    public WatchContracts.WatchHeartRateSnapshot snapshot(double nowMonotonicSec) {
        synchronized (lock) {
            pruneLocked(nowMonotonicSec);
            WatchContracts.WatchConnectionStatus effective = status;
            if (effective == WatchContracts.WatchConnectionStatus.STREAMING
                    && !samples.isEmpty()
                    && nowMonotonicSec - samples.get(samples.size() - 1).receivedMonotonicSec
                            > STALE_AFTER_SEC) {
                effective = WatchContracts.WatchConnectionStatus.STALE;
            }
            return new WatchContracts.WatchHeartRateSnapshot(
                    effective,
                    new ArrayList<>(samples),
                    new ArrayList<>(devices),
                    errorCode,
                    malformedPackets,
                    reconnectAttempts);
        }
    }

    private void pruneLocked(double nowMonotonicSec) {
        double cutoff = nowMonotonicSec - HISTORY_SEC;
        samples.removeIf(sample -> sample.receivedMonotonicSec < cutoff);
    }
}

package com.jagger.rppgbench.watch;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.DoubleSupplier;

public final class WatchBleWorker implements AutoCloseable {
    public interface Sleeper {
        void sleep(double seconds) throws InterruptedException;
    }

    private static final double[] DEFAULT_RECONNECT_DELAYS = {0.25, 0.5, 1.0};

    private final WatchBleBackend backend;
    private final DoubleSupplier clock;
    private final Sleeper sleeper;
    private final int maxReconnectAttempts;
    private final double[] reconnectDelays;
    private final WatchSampleStore store = new WatchSampleStore();
    private final ExecutorService executor;
    private final Object stateLock = new Object();
    private final Map<String, WatchBleBackend.DeviceHandle> handles = new HashMap<>();
    private final AtomicBoolean userDisconnect = new AtomicBoolean(false);
    private final AtomicBoolean reconnectInFlight = new AtomicBoolean(false);
    private String selectedDeviceId;
    private String selectedDeviceName = "";
    private boolean closed;

    public WatchBleWorker(WatchBleBackend backend) {
        this(
                backend,
                () -> System.nanoTime() / 1_000_000_000.0,
                seconds -> Thread.sleep(Math.max(0L, Math.round(seconds * 1000.0))),
                3,
                DEFAULT_RECONNECT_DELAYS);
    }

    public WatchBleWorker(
            WatchBleBackend backend,
            DoubleSupplier clock,
            Sleeper sleeper,
            int maxReconnectAttempts,
            double[] reconnectDelays) {
        this.backend = Objects.requireNonNull(backend, "backend");
        this.clock = Objects.requireNonNull(clock, "clock");
        this.sleeper = Objects.requireNonNull(sleeper, "sleeper");
        this.maxReconnectAttempts = Math.max(0, maxReconnectAttempts);
        this.reconnectDelays =
                reconnectDelays == null ? DEFAULT_RECONNECT_DELAYS.clone() : reconnectDelays.clone();
        this.executor =
                Executors.newSingleThreadExecutor(
                        r -> {
                            Thread thread = new Thread(r, "watch-ble-worker");
                            thread.setDaemon(true);
                            return thread;
                        });
    }

    public boolean startScan(double timeoutSec) {
        WatchContracts.WatchConnectionStatus status = store.snapshot(now()).status;
        if (status == WatchContracts.WatchConnectionStatus.SCANNING
                || status == WatchContracts.WatchConnectionStatus.CONNECTING) {
            return false;
        }
        store.setStatus(WatchContracts.WatchConnectionStatus.SCANNING);
        return enqueue(() -> doScan(Math.max(0.0, timeoutSec)));
    }

    public boolean connect(String deviceId) {
        WatchContracts.WatchConnectionStatus status = store.snapshot(now()).status;
        if (status == WatchContracts.WatchConnectionStatus.CONNECTING
                || status == WatchContracts.WatchConnectionStatus.RECONNECTING
                || status == WatchContracts.WatchConnectionStatus.STREAMING) {
            return false;
        }
        WatchBleBackend.DeviceHandle handle;
        synchronized (stateLock) {
            handle = handles.get(deviceId);
        }
        if (handle == null) {
            store.setError("DEVICE_NOT_FOUND");
            return false;
        }
        userDisconnect.set(false);
        synchronized (stateLock) {
            selectedDeviceId = handle.id();
            selectedDeviceName = handle.name();
        }
        store.setReconnectAttempts(0);
        store.setStatus(WatchContracts.WatchConnectionStatus.CONNECTING);
        return enqueue(() -> doConnect(handle));
    }

    public void disconnect() {
        userDisconnect.set(true);
        enqueue(this::doDisconnect);
    }

    public WatchContracts.WatchHeartRateSnapshot snapshot(double nowMonotonicSec) {
        return store.snapshot(nowMonotonicSec);
    }

    @Override
    public void close() {
        userDisconnect.set(true);
        synchronized (stateLock) {
            if (closed) {
                return;
            }
            closed = true;
        }
        try {
            executor.submit(this::doDisconnect).get(2, TimeUnit.SECONDS);
        } catch (Exception ignored) {
            // Best-effort shutdown.
        }
        executor.shutdownNow();
        try {
            executor.awaitTermination(2, TimeUnit.SECONDS);
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        }
        store.setReconnectAttempts(0);
        store.setStatus(WatchContracts.WatchConnectionStatus.DISCONNECTED);
    }

    private boolean enqueue(Runnable work) {
        synchronized (stateLock) {
            if (closed) {
                return false;
            }
        }
        try {
            executor.execute(work);
            return true;
        } catch (RejectedExecutionException rejected) {
            return false;
        }
    }

    private void doScan(double timeoutSec) {
        try {
            List<WatchBleBackend.DeviceHandle> found = backend.scan(timeoutSec);
            List<WatchContracts.WatchDevice> devices = new ArrayList<>();
            synchronized (stateLock) {
                handles.clear();
                for (WatchBleBackend.DeviceHandle handle : found) {
                    handles.put(handle.id(), handle);
                    devices.add(new WatchContracts.WatchDevice(handle.id(), handle.name()));
                }
            }
            store.setDevices(devices);
            store.setStatus(WatchContracts.WatchConnectionStatus.DISCONNECTED);
        } catch (Exception exc) {
            store.setError(errorCode(exc));
        }
    }

    private void doConnect(WatchBleBackend.DeviceHandle handle) {
        try {
            backend.connect(handle, this::onNotification, this::onDisconnect);
            store.setReconnectAttempts(0);
            store.setStatus(WatchContracts.WatchConnectionStatus.STREAMING);
        } catch (Exception exc) {
            store.setError(errorCode(exc));
        }
    }

    private void doDisconnect() {
        try {
            backend.disconnect();
        } catch (Exception ignored) {
            // Always clear local connection state.
        } finally {
            store.setReconnectAttempts(0);
            store.setStatus(WatchContracts.WatchConnectionStatus.DISCONNECTED);
        }
    }

    private void onNotification(byte[] payload) {
        try {
            HeartRateParser.Parsed parsed = HeartRateParser.parse(payload);
            String deviceId;
            String deviceName;
            synchronized (stateLock) {
                deviceId = selectedDeviceId == null ? "" : selectedDeviceId;
                deviceName = selectedDeviceName;
            }
            store.acceptMeasurement(now(), parsed.bpm, parsed.rrIntervalsSec, deviceId, deviceName);
        } catch (IllegalArgumentException | NullPointerException malformed) {
            store.incrementMalformed();
        }
    }

    private void onDisconnect() {
        if (userDisconnect.get()) {
            return;
        }
        if (!reconnectInFlight.compareAndSet(false, true)) {
            return;
        }
        enqueue(
                () -> {
                    try {
                        doReconnect();
                    } finally {
                        reconnectInFlight.set(false);
                    }
                });
    }

    private void doReconnect() {
        WatchBleBackend.DeviceHandle handle;
        synchronized (stateLock) {
            handle = selectedDeviceId == null ? null : handles.get(selectedDeviceId);
        }
        if (handle == null) {
            store.setError("DEVICE_NOT_FOUND");
            return;
        }
        int attempts = Math.min(maxReconnectAttempts, reconnectDelays.length);
        for (int attempt = 1; attempt <= attempts; attempt++) {
            if (userDisconnect.get()) {
                return;
            }
            store.setReconnectAttempts(attempt);
            store.setStatus(WatchContracts.WatchConnectionStatus.RECONNECTING);
            try {
                sleeper.sleep(Math.max(0.0, reconnectDelays[attempt - 1]));
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return;
            }
            if (userDisconnect.get()) {
                return;
            }
            try {
                backend.connect(handle, this::onNotification, this::onDisconnect);
                store.setStatus(WatchContracts.WatchConnectionStatus.STREAMING);
                return;
            } catch (Exception exc) {
                String code = errorCode(exc);
                if ("INCOMPATIBLE_DEVICE".equals(code)) {
                    store.setReconnectAttempts(attempt);
                    store.setError(code);
                    return;
                }
            }
        }
        store.setReconnectAttempts(attempts);
        store.setError("RECONNECT_FAILED");
    }

    private double now() {
        return clock.getAsDouble();
    }

    static String errorCode(Exception exc) {
        String text = String.valueOf(exc.getMessage());
        if (text == null) {
            text = "";
        }
        text = text.toUpperCase(Locale.ROOT);
        if (text.contains("INCOMPATIBLE_DEVICE")) {
            return "INCOMPATIBLE_DEVICE";
        }
        if (text.contains("NOT AVAILABLE")
                || text.contains("PERMISSION")
                || text.contains("UNAUTHORIZED")) {
            return "BLUETOOTH_PERMISSION_DENIED";
        }
        return "BLE_OPERATION_FAILED";
    }
}

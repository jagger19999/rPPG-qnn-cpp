package com.jagger.rppgbench.watch;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import org.junit.Test;

public class WatchBleWorkerTest {
    private static final class FakeDevice implements WatchBleBackend.DeviceHandle {
        private final String id;
        private final String name;

        FakeDevice(String id, String name) {
            this.id = id;
            this.name = name;
        }

        @Override
        public String id() {
            return id;
        }

        @Override
        public String name() {
            return name;
        }
    }

    private static final class FakeBleBackend implements WatchBleBackend {
        final AtomicInteger connectCalls = new AtomicInteger();
        final AtomicInteger disconnectCalls = new AtomicInteger();
        final AtomicReference<Consumer<byte[]>> notifications = new AtomicReference<>();
        final AtomicReference<Runnable> disconnectCallback = new AtomicReference<>();
        volatile boolean failConnect;
        volatile Exception scanError;
        volatile Exception connectError;
        final List<Double> slept = new CopyOnWriteArrayList<>();

        @Override
        public List<DeviceHandle> scan(double timeoutSec) throws Exception {
            if (scanError != null) {
                throw scanError;
            }
            return Collections.singletonList(new FakeDevice("dev-1", "HUAWEI WATCH GT 5 Pro"));
        }

        @Override
        public void connect(DeviceHandle handle, Consumer<byte[]> onNotification, Runnable onDisconnect)
                throws Exception {
            connectCalls.incrementAndGet();
            if (failConnect) {
                throw new RuntimeException("link lost");
            }
            if (connectError != null) {
                throw connectError;
            }
            notifications.set(onNotification);
            disconnectCallback.set(onDisconnect);
        }

        @Override
        public void disconnect() {
            disconnectCalls.incrementAndGet();
        }
    }

    private static void waitUntil(CheckedBooleanSupplier predicate) throws Exception {
        long deadline = System.nanoTime() + 2_000_000_000L;
        while (System.nanoTime() < deadline) {
            if (predicate.getAsBoolean()) {
                return;
            }
            Thread.sleep(10);
        }
        throw new AssertionError("condition not reached");
    }

    private interface CheckedBooleanSupplier {
        boolean getAsBoolean() throws Exception;
    }

    private static WatchBleWorker connectedWorker(FakeBleBackend backend) throws Exception {
        WatchBleWorker worker =
                new WatchBleWorker(
                        backend,
                        () -> System.nanoTime() / 1_000_000_000.0,
                        seconds -> {
                            backend.slept.add(seconds);
                        },
                        3,
                        new double[] {0.0, 0.0, 0.0});
        assertTrue(worker.startScan(0.01));
        waitUntil(() -> !worker.snapshot(0).devices.isEmpty());
        assertTrue(worker.connect("dev-1"));
        waitUntil(() -> backend.notifications.get() != null);
        waitUntil(
                () ->
                        worker.snapshot(System.nanoTime() / 1_000_000_000.0).status
                                == WatchContracts.WatchConnectionStatus.STREAMING);
        return worker;
    }

    @Test
    public void scanConnectAndNotificationAreNonBlocking() throws Exception {
        FakeBleBackend backend = new FakeBleBackend();
        WatchBleWorker worker = connectedWorker(backend);
        try {
            backend.notifications.get().accept(new byte[] {0, 73});
            waitUntil(() -> worker.snapshot(System.nanoTime() / 1_000_000_000.0).latestSample != null);

            WatchContracts.WatchHeartRateSnapshot snap =
                    worker.snapshot(System.nanoTime() / 1_000_000_000.0);
            assertNotNull(snap.latestSample);
            assertEquals(73, snap.latestSample.bpm);
            assertEquals(WatchContracts.WatchConnectionStatus.STREAMING, snap.status);
            assertEquals("dev-1", snap.latestSample.deviceId);
        } finally {
            worker.close();
        }
    }

    @Test
    public void malformedPacketIsCountedWithoutStoppingStream() throws Exception {
        FakeBleBackend backend = new FakeBleBackend();
        WatchBleWorker worker = connectedWorker(backend);
        try {
            backend.notifications.get().accept(new byte[0]);
            waitUntil(
                    () ->
                            worker.snapshot(System.nanoTime() / 1_000_000_000.0).malformedPackets
                                    == 1);

            WatchContracts.WatchHeartRateSnapshot snap =
                    worker.snapshot(System.nanoTime() / 1_000_000_000.0);
            assertEquals(1, snap.malformedPackets);
            assertEquals(WatchContracts.WatchConnectionStatus.STREAMING, snap.status);
        } finally {
            worker.close();
        }
    }

    @Test
    public void unexpectedDisconnectRetriesThreeTimesThenErrors() throws Exception {
        FakeBleBackend backend = new FakeBleBackend();
        WatchBleWorker worker = connectedWorker(backend);
        try {
            backend.failConnect = true;
            backend.disconnectCallback.get().run();
            waitUntil(
                    () ->
                            worker.snapshot(System.nanoTime() / 1_000_000_000.0).status
                                    == WatchContracts.WatchConnectionStatus.ERROR);

            assertEquals(4, backend.connectCalls.get());
            WatchContracts.WatchHeartRateSnapshot snap =
                    worker.snapshot(System.nanoTime() / 1_000_000_000.0);
            assertEquals(3, snap.reconnectAttempts);
            assertEquals("RECONNECT_FAILED", snap.errorCode);
            assertEquals(3, backend.slept.size());
        } finally {
            worker.close();
        }
    }

    @Test
    public void incompatibleDeviceStopsReconnect() throws Exception {
        FakeBleBackend backend = new FakeBleBackend();
        WatchBleWorker worker = connectedWorker(backend);
        try {
            backend.connectError = new RuntimeException("INCOMPATIBLE_DEVICE missing HR service");
            backend.disconnectCallback.get().run();
            waitUntil(
                    () ->
                            worker.snapshot(System.nanoTime() / 1_000_000_000.0).status
                                    == WatchContracts.WatchConnectionStatus.ERROR);

            assertEquals(2, backend.connectCalls.get());
            WatchContracts.WatchHeartRateSnapshot snap =
                    worker.snapshot(System.nanoTime() / 1_000_000_000.0);
            assertEquals("INCOMPATIBLE_DEVICE", snap.errorCode);
            assertEquals(1, snap.reconnectAttempts);
        } finally {
            worker.close();
        }
    }

    @Test
    public void explicitDisconnectSuppressesReconnect() throws Exception {
        FakeBleBackend backend = new FakeBleBackend();
        WatchBleWorker worker = connectedWorker(backend);
        try {
            Runnable callback = backend.disconnectCallback.get();
            worker.disconnect();
            waitUntil(
                    () ->
                            worker.snapshot(System.nanoTime() / 1_000_000_000.0).status
                                    == WatchContracts.WatchConnectionStatus.DISCONNECTED);
            callback.run();
            Thread.sleep(50);

            assertEquals(1, backend.connectCalls.get());
            assertFalse(
                    worker.snapshot(System.nanoTime() / 1_000_000_000.0).status
                            == WatchContracts.WatchConnectionStatus.RECONNECTING);
        } finally {
            worker.close();
        }
    }

    @Test
    public void scanPermissionErrorIsExposedWithoutRaising() throws Exception {
        FakeBleBackend backend = new FakeBleBackend();
        backend.scanError = new RuntimeException("Bluetooth permission unauthorized");
        WatchBleWorker worker =
                new WatchBleWorker(
                        backend,
                        () -> System.nanoTime() / 1_000_000_000.0,
                        seconds -> {},
                        3,
                        new double[] {0.0, 0.0, 0.0});
        try {
            assertTrue(worker.startScan(0.01));
            waitUntil(
                    () ->
                            worker.snapshot(System.nanoTime() / 1_000_000_000.0).status
                                    == WatchContracts.WatchConnectionStatus.ERROR);

            assertEquals(
                    "BLUETOOTH_PERMISSION_DENIED",
                    worker.snapshot(System.nanoTime() / 1_000_000_000.0).errorCode);
        } finally {
            worker.close();
        }
    }
}

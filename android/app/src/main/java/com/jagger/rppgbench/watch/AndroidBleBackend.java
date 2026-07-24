package com.jagger.rppgbench.watch;

import android.annotation.SuppressLint;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattDescriptor;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothProfile;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.os.Build;
import android.os.ParcelUuid;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/**
 * Android BLE adapter for standard Heart Rate Service notifications.
 * Must be driven from {@link WatchBleWorker}'s executor, not Camera2 callbacks.
 */
public final class AndroidBleBackend implements WatchBleBackend {
    public static final UUID HEART_RATE_SERVICE =
            UUID.fromString("0000180d-0000-1000-8000-00805f9b34fb");
    public static final UUID HEART_RATE_MEASUREMENT =
            UUID.fromString("00002a37-0000-1000-8000-00805f9b34fb");
    public static final UUID CLIENT_CHARACTERISTIC_CONFIG =
            UUID.fromString("00002902-0000-1000-8000-00805f9b34fb");

    private final Context context;
    private final Object lock = new Object();

    private BluetoothGatt gatt;
    private Consumer<byte[]> notificationConsumer;
    private Runnable disconnectRunnable;
    private boolean userClosed;

    public AndroidBleBackend(Context context) {
        this.context = Objects.requireNonNull(context, "context").getApplicationContext();
    }

    @Override
    @SuppressLint("MissingPermission")
    public List<DeviceHandle> scan(double timeoutSec) throws Exception {
        BluetoothAdapter adapter = requireAdapter();
        BluetoothLeScanner scanner = adapter.getBluetoothLeScanner();
        if (scanner == null) {
            throw new IllegalStateException("Bluetooth LE scanner is not available");
        }

        Map<String, DeviceHandle> discovered = new LinkedHashMap<>();
        CountDownLatch done = new CountDownLatch(1);
        ScanCallback callback =
                new ScanCallback() {
                    @Override
                    public void onScanResult(int callbackType, ScanResult result) {
                        maybeAdd(discovered, result);
                    }

                    @Override
                    public void onBatchScanResults(List<ScanResult> results) {
                        for (ScanResult result : results) {
                            maybeAdd(discovered, result);
                        }
                    }

                    @Override
                    public void onScanFailed(int errorCode) {
                        done.countDown();
                    }
                };

        ScanSettings settings =
                new ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build();
        scanner.startScan(null, settings, callback);
        long waitMs = Math.max(0L, Math.round(timeoutSec * 1000.0));
        try {
            done.await(waitMs, TimeUnit.MILLISECONDS);
        } finally {
            try {
                scanner.stopScan(callback);
            } catch (SecurityException | IllegalStateException ignored) {
                // Best-effort stop.
            }
        }
        return new ArrayList<>(discovered.values());
    }

    @Override
    @SuppressLint("MissingPermission")
    public void connect(
            DeviceHandle handle, Consumer<byte[]> onNotification, Runnable onDisconnect)
            throws Exception {
        Objects.requireNonNull(handle, "handle");
        Objects.requireNonNull(onNotification, "onNotification");
        Objects.requireNonNull(onDisconnect, "onDisconnect");

        BluetoothDevice device;
        if (handle instanceof AndroidDevice) {
            device = ((AndroidDevice) handle).device;
        } else {
            device = requireAdapter().getRemoteDevice(handle.id());
        }

        synchronized (lock) {
            closeGattLocked();
            notificationConsumer = onNotification;
            disconnectRunnable = onDisconnect;
            userClosed = false;
        }

        CountDownLatch ready = new CountDownLatch(1);
        AtomicReference<Exception> failure = new AtomicReference<>();

        BluetoothGattCallback callback =
                new BluetoothGattCallback() {
                    @Override
                    public void onConnectionStateChange(BluetoothGatt g, int status, int newState) {
                        if (newState == BluetoothProfile.STATE_CONNECTED) {
                            if (!g.discoverServices()) {
                                failure.compareAndSet(
                                        null, new IllegalStateException("BLE service discovery failed"));
                                ready.countDown();
                            }
                            return;
                        }
                        if (newState == BluetoothProfile.STATE_DISCONNECTED) {
                            boolean closedByUser;
                            Runnable disconnectCb;
                            synchronized (lock) {
                                closedByUser = userClosed;
                                disconnectCb = disconnectRunnable;
                                if (gatt == g) {
                                    gatt = null;
                                }
                            }
                            try {
                                g.close();
                            } catch (Exception ignored) {
                            }
                            if (!closedByUser && disconnectCb != null && ready.getCount() == 0) {
                                disconnectCb.run();
                            }
                            if (ready.getCount() > 0) {
                                failure.compareAndSet(
                                        null,
                                        new IllegalStateException(
                                                status == BluetoothGatt.GATT_SUCCESS
                                                        ? "BLE disconnected before ready"
                                                        : "BLE connect failed status=" + status));
                                ready.countDown();
                            }
                        }
                    }

                    @Override
                    public void onServicesDiscovered(BluetoothGatt g, int status) {
                        if (status != BluetoothGatt.GATT_SUCCESS) {
                            failure.compareAndSet(
                                    null, new IllegalStateException("BLE service discovery status=" + status));
                            ready.countDown();
                            return;
                        }
                        BluetoothGattService service = g.getService(HEART_RATE_SERVICE);
                        if (service == null) {
                            failure.compareAndSet(
                                    null, new IllegalStateException("INCOMPATIBLE_DEVICE"));
                            g.disconnect();
                            ready.countDown();
                            return;
                        }
                        BluetoothGattCharacteristic characteristic =
                                service.getCharacteristic(HEART_RATE_MEASUREMENT);
                        if (characteristic == null) {
                            failure.compareAndSet(
                                    null, new IllegalStateException("INCOMPATIBLE_DEVICE"));
                            g.disconnect();
                            ready.countDown();
                            return;
                        }
                        if (!g.setCharacteristicNotification(characteristic, true)) {
                            failure.compareAndSet(
                                    null, new IllegalStateException("BLE notification enable failed"));
                            ready.countDown();
                            return;
                        }
                        BluetoothGattDescriptor cccd =
                                characteristic.getDescriptor(CLIENT_CHARACTERISTIC_CONFIG);
                        if (cccd == null) {
                            failure.compareAndSet(
                                    null, new IllegalStateException("INCOMPATIBLE_DEVICE"));
                            g.disconnect();
                            ready.countDown();
                            return;
                        }
                        boolean wrote;
                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                            int writeStatus =
                                    g.writeDescriptor(
                                            cccd, BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                            wrote = writeStatus == android.bluetooth.BluetoothStatusCodes.SUCCESS;
                        } else {
                            cccd.setValue(BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE);
                            wrote = g.writeDescriptor(cccd);
                        }
                        if (!wrote) {
                            failure.compareAndSet(
                                    null, new IllegalStateException("BLE CCCD write failed"));
                            ready.countDown();
                        }
                    }

                    @Override
                    public void onDescriptorWrite(
                            BluetoothGatt g, BluetoothGattDescriptor descriptor, int status) {
                        if (status != BluetoothGatt.GATT_SUCCESS) {
                            failure.compareAndSet(
                                    null, new IllegalStateException("BLE CCCD write status=" + status));
                        }
                        ready.countDown();
                    }

                    @Override
                    public void onCharacteristicChanged(
                            BluetoothGatt g, BluetoothGattCharacteristic characteristic, byte[] value) {
                        deliverNotification(value);
                    }

                    @Override
                    @Deprecated
                    public void onCharacteristicChanged(
                            BluetoothGatt g, BluetoothGattCharacteristic characteristic) {
                        byte[] value = characteristic.getValue();
                        deliverNotification(value == null ? new byte[0] : value);
                    }
                };

        BluetoothGatt opened =
                Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                        ? device.connectGatt(context, false, callback, BluetoothDevice.TRANSPORT_LE)
                        : device.connectGatt(context, false, callback);
        if (opened == null) {
            throw new IllegalStateException("BLE connectGatt returned null");
        }
        synchronized (lock) {
            gatt = opened;
        }

        if (!ready.await(20, TimeUnit.SECONDS)) {
            disconnect();
            throw new IllegalStateException("BLE connect timed out");
        }
        Exception error = failure.get();
        if (error != null) {
            disconnect();
            throw error;
        }
    }

    @Override
    @SuppressLint("MissingPermission")
    public void disconnect() throws Exception {
        synchronized (lock) {
            userClosed = true;
            closeGattLocked();
            notificationConsumer = null;
            disconnectRunnable = null;
        }
    }

    private void deliverNotification(byte[] value) {
        Consumer<byte[]> consumer;
        synchronized (lock) {
            consumer = notificationConsumer;
        }
        if (consumer != null) {
            consumer.accept(value == null ? new byte[0] : value);
        }
    }

    @SuppressLint("MissingPermission")
    private void closeGattLocked() {
        BluetoothGatt current = gatt;
        gatt = null;
        if (current != null) {
            try {
                current.disconnect();
            } catch (Exception ignored) {
            }
            try {
                current.close();
            } catch (Exception ignored) {
            }
        }
    }

    private BluetoothAdapter requireAdapter() {
        BluetoothManager manager =
                (BluetoothManager) context.getSystemService(Context.BLUETOOTH_SERVICE);
        if (manager == null) {
            throw new IllegalStateException("BluetoothManager is not available");
        }
        BluetoothAdapter adapter = manager.getAdapter();
        if (adapter == null || !adapter.isEnabled()) {
            throw new IllegalStateException("Bluetooth adapter is not available");
        }
        return adapter;
    }

    private static void maybeAdd(Map<String, DeviceHandle> discovered, ScanResult result) {
        if (result == null || result.getDevice() == null) {
            return;
        }
        BluetoothDevice device = result.getDevice();
        String address = device.getAddress();
        if (address == null || discovered.containsKey(address)) {
            return;
        }
        String name = result.getScanRecord() != null ? result.getScanRecord().getDeviceName() : null;
        if (name == null || name.isEmpty()) {
            name = device.getName();
        }
        if (name == null || name.isEmpty()) {
            name = "Unknown heart-rate device";
        }
        boolean matchesService = false;
        if (result.getScanRecord() != null && result.getScanRecord().getServiceUuids() != null) {
            for (ParcelUuid uuid : result.getScanRecord().getServiceUuids()) {
                if (uuid != null && HEART_RATE_SERVICE.equals(uuid.getUuid())) {
                    matchesService = true;
                    break;
                }
            }
        }
        String normalized = name.toLowerCase(Locale.ROOT);
        if (!matchesService && !normalized.contains("huawei") && !normalized.contains("heart")) {
            return;
        }
        discovered.put(address, new AndroidDevice(device, name));
    }

    private static final class AndroidDevice implements DeviceHandle {
        final BluetoothDevice device;
        private final String name;

        AndroidDevice(BluetoothDevice device, String name) {
            this.device = device;
            this.name = name;
        }

        @Override
        public String id() {
            return device.getAddress();
        }

        @Override
        public String name() {
            return name;
        }
    }
}

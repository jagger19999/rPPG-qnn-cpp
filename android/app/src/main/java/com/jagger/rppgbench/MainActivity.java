package com.jagger.rppgbench;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;

import com.jagger.rppgbench.watch.AndroidBleBackend;
import com.jagger.rppgbench.watch.WatchAligner;
import com.jagger.rppgbench.watch.WatchBleWorker;
import com.jagger.rppgbench.watch.WatchContracts;
import com.jagger.rppgbench.watch.WatchCsvExport;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class MainActivity extends Activity {
    private static final int CAMERA_PERMISSION_REQUEST = 1001;
    private static final int BLE_PERMISSION_REQUEST = 1002;
    private static final int ACTION_NONE = 0;
    private static final int ACTION_LIST = 1;
    private static final int ACTION_START = 2;
    private static final int BLE_ACTION_NONE = 0;
    private static final int BLE_ACTION_SCAN = 1;
    private static final int BLE_ACTION_CONNECT = 2;

    private final Handler statusHandler = new Handler(Looper.getMainLooper());
    private final Runnable statusPoll =
            new Runnable() {
                @Override
                public void run() {
                    refreshCombinedStatus();
                    statusHandler.postDelayed(this, 1000);
                }
            };

    private TextView status;
    private TextView watchStatusView;
    private Spinner methodSelector;
    private Spinner watchDeviceSelector;
    private CheckBox deepSelector;
    private ArrayAdapter<String> watchDeviceAdapter;
    private final List<WatchContracts.WatchDevice> watchDevices = new ArrayList<>();
    private final List<WatchContracts.WatchAlignmentResult> alignmentHistory = new ArrayList<>();

    private String cameraId;
    private long nativeHandle;
    private boolean started;
    private int pendingAction = ACTION_NONE;
    private int pendingBleAction = BLE_ACTION_NONE;
    private WatchBleWorker watchWorker;
    private File sessionOutputDirectory;
    private Double sessionStartMonotonicSec;
    private Double lastAlignedWindowEndSec;
    private WatchContracts.WatchAlignmentResult latestAlignment;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        int padding = Math.round(24 * getResources().getDisplayMetrics().density);
        content.setPadding(padding, padding, padding, padding);

        TextView title = new TextView(this);
        title.setText(R.string.app_title);
        content.addView(title);

        TextView boundary = new TextView(this);
        boundary.setText(R.string.camera_smoke_boundary);
        content.addView(boundary);

        status = new TextView(this);
        content.addView(status);

        methodSelector = new Spinner(this);
        ArrayAdapter<String> methodAdapter =
                new ArrayAdapter<>(
                        this,
                        android.R.layout.simple_spinner_item,
                        new String[] {"green", "pos", "chrom"});
        methodAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        methodSelector.setAdapter(methodAdapter);
        content.addView(methodSelector);

        deepSelector = new CheckBox(this);
        deepSelector.setText("Run EfficientPhys with ONNX Runtime CPU");
        content.addView(deepSelector);

        Button list = new Button(this);
        list.setText(R.string.list_cameras);
        list.setOnClickListener(view -> runWithCameraPermission(ACTION_LIST));
        content.addView(list);

        Button start = new Button(this);
        start.setText(R.string.start_camera);
        start.setOnClickListener(view -> runWithCameraPermission(ACTION_START));
        content.addView(start);

        Button stop = new Button(this);
        stop.setText(R.string.stop_camera);
        stop.setOnClickListener(view -> stopCamera());
        content.addView(stop);

        TextView disclaimer = new TextView(this);
        disclaimer.setText(R.string.watch_disclaimer);
        content.addView(disclaimer);

        watchStatusView = new TextView(this);
        content.addView(watchStatusView);

        watchDeviceAdapter =
                new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, new ArrayList<>());
        watchDeviceAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        watchDeviceSelector = new Spinner(this);
        watchDeviceSelector.setAdapter(watchDeviceAdapter);
        content.addView(watchDeviceSelector);

        Button scanWatch = new Button(this);
        scanWatch.setText(R.string.scan_watch);
        scanWatch.setOnClickListener(view -> runWithBlePermission(BLE_ACTION_SCAN));
        content.addView(scanWatch);

        Button connectWatch = new Button(this);
        connectWatch.setText(R.string.connect_watch);
        connectWatch.setOnClickListener(view -> runWithBlePermission(BLE_ACTION_CONNECT));
        content.addView(connectWatch);

        Button disconnectWatch = new Button(this);
        disconnectWatch.setText(R.string.disconnect_watch);
        disconnectWatch.setOnClickListener(view -> disconnectWatch());
        content.addView(disconnectWatch);

        setContentView(content);
        ensureWatchWorker();
        refreshNativeStatus();
        statusHandler.post(statusPoll);
    }

    private void ensureWatchWorker() {
        if (watchWorker != null) {
            return;
        }
        watchWorker =
                new WatchBleWorker(
                        new AndroidBleBackend(this),
                        MainActivity::monotonicSec,
                        seconds -> Thread.sleep(Math.max(0L, Math.round(seconds * 1000.0))),
                        3,
                        new double[] {0.25, 0.5, 1.0});
    }

    private static double monotonicSec() {
        return SystemClock.elapsedRealtimeNanos() / 1_000_000_000.0;
    }

    private void refreshNativeStatus() {
        try {
            status.setText(NativeBridge.nativeBuildIdentity());
        } catch (Throwable error) {
            status.setText("NATIVE_LOAD_FAILED: " + error.getClass().getSimpleName());
        }
        refreshWatchLabels();
    }

    private void refreshCombinedStatus() {
        StringBuilder builder = new StringBuilder();
        if (started && nativeHandle != 0) {
            try {
                String cameraStatus = NativeBridge.nativeGetStatus(nativeHandle);
                builder.append(cameraStatus);
                maybeAlignFromCameraStatus(cameraStatus);
            } catch (Throwable error) {
                builder.append("CAMERA_STATUS_FAILED: ").append(error.getMessage());
            }
        } else {
            try {
                builder.append(NativeBridge.nativeBuildIdentity());
            } catch (Throwable error) {
                builder.append("NATIVE_LOAD_FAILED: ").append(error.getClass().getSimpleName());
            }
        }
        builder.append('\n').append(formatWatchStatusLine());
        status.setText(builder.toString());
        refreshWatchLabels();
        refreshWatchDeviceSpinner();
    }

    private void maybeAlignFromCameraStatus(String cameraStatusJson) {
        if (watchWorker == null || sessionStartMonotonicSec == null) {
            return;
        }
        try {
            JSONObject json = new JSONObject(cameraStatusJson);
            if (!json.optBoolean("heart_rate_available", false)) {
                return;
            }
            double windowStart = json.optDouble("window_start_sec", Double.NaN);
            double windowEnd = json.optDouble("window_end_sec", Double.NaN);
            if (!Double.isFinite(windowStart) || !Double.isFinite(windowEnd) || windowEnd <= windowStart) {
                return;
            }
            if (lastAlignedWindowEndSec != null && Double.compare(lastAlignedWindowEndSec, windowEnd) == 0) {
                return;
            }
            boolean valid = json.optBoolean("heart_rate_valid", false);
            Double bpm = json.has("bpm") ? json.optDouble("bpm") : null;
            WatchAligner.RppgWindow window =
                    new WatchAligner.RppgWindow(windowStart, windowEnd, bpm, valid);
            WatchContracts.WatchHeartRateSnapshot snapshot =
                    watchWorker.snapshot(monotonicSec());
            WatchContracts.WatchAlignmentResult alignment =
                    WatchAligner.align(window, snapshot, sessionStartMonotonicSec);
            alignmentHistory.add(alignment);
            latestAlignment = alignment;
            lastAlignedWindowEndSec = windowEnd;
        } catch (Exception ignored) {
            // Keep camera status visible even if watch merge fails.
        }
    }

    private String formatWatchStatusLine() {
        if (watchWorker == null) {
            return "watch_status=DISCONNECTED";
        }
        WatchContracts.WatchHeartRateSnapshot snapshot = watchWorker.snapshot(monotonicSec());
        StringBuilder line = new StringBuilder();
        line.append("watch_status=").append(snapshot.status.name());
        if (snapshot.latestSample != null) {
            line.append("\nwatch_bpm=").append(snapshot.latestSample.bpm);
        }
        if (snapshot.errorCode != null) {
            line.append("\nwatch_error=").append(snapshot.errorCode);
        }
        if (latestAlignment != null) {
            line.append("\nwatch_alignment=").append(latestAlignment.status.name());
            if (latestAlignment.watchReferenceBpm != null) {
                line.append("\nwatch_reference_bpm=")
                        .append(formatDouble(latestAlignment.watchReferenceBpm));
            }
            if (latestAlignment.absoluteErrorBpm != null) {
                line.append("\nwatch_abs_error_bpm=")
                        .append(formatDouble(latestAlignment.absoluteErrorBpm));
            }
            line.append("\nwatch_coverage=").append(formatDouble(latestAlignment.coverageRatio));
        }
        return line.toString();
    }

    private void refreshWatchLabels() {
        if (watchStatusView == null) {
            return;
        }
        watchStatusView.setText(formatWatchStatusLine());
    }

    private void refreshWatchDeviceSpinner() {
        if (watchWorker == null || watchDeviceAdapter == null) {
            return;
        }
        WatchContracts.WatchHeartRateSnapshot snapshot = watchWorker.snapshot(monotonicSec());
        if (sameDevices(watchDevices, snapshot.devices)) {
            return;
        }
        watchDevices.clear();
        watchDevices.addAll(snapshot.devices);
        watchDeviceAdapter.clear();
        for (WatchContracts.WatchDevice device : watchDevices) {
            watchDeviceAdapter.add(device.name + " (" + device.id + ")");
        }
        watchDeviceAdapter.notifyDataSetChanged();
    }

    private static boolean sameDevices(
            List<WatchContracts.WatchDevice> left, List<WatchContracts.WatchDevice> right) {
        if (left.size() != right.size()) {
            return false;
        }
        for (int index = 0; index < left.size(); index++) {
            WatchContracts.WatchDevice a = left.get(index);
            WatchContracts.WatchDevice b = right.get(index);
            if (!a.id.equals(b.id) || !a.name.equals(b.name)) {
                return false;
            }
        }
        return true;
    }

    private void runWithCameraPermission(int action) {
        if (checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            pendingAction = action;
            requestPermissions(
                    new String[] {Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
            return;
        }
        performAction(action);
    }

    private void runWithBlePermission(int action) {
        String[] needed = blePermissions();
        boolean missing = false;
        for (String permission : needed) {
            if (checkSelfPermission(permission) != PackageManager.PERMISSION_GRANTED) {
                missing = true;
                break;
            }
        }
        if (missing) {
            pendingBleAction = action;
            requestPermissions(needed, BLE_PERMISSION_REQUEST);
            return;
        }
        performBleAction(action);
    }

    private static String[] blePermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            return new String[] {
                Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT
            };
        }
        return new String[] {
            Manifest.permission.BLUETOOTH, Manifest.permission.BLUETOOTH_ADMIN
        };
    }

    private void performAction(int action) {
        if (action == ACTION_LIST) {
            listCameras();
        } else if (action == ACTION_START) {
            startCamera();
        }
    }

    private void performBleAction(int action) {
        if (action == BLE_ACTION_SCAN) {
            scanWatch();
        } else if (action == BLE_ACTION_CONNECT) {
            connectWatch();
        }
    }

    private void scanWatch() {
        ensureWatchWorker();
        if (!watchWorker.startScan(15.0)) {
            status.setText("WATCH_SCAN_BUSY");
            return;
        }
        refreshCombinedStatus();
    }

    private void connectWatch() {
        ensureWatchWorker();
        int index = watchDeviceSelector.getSelectedItemPosition();
        if (index < 0 || index >= watchDevices.size()) {
            status.setText("WATCH_DEVICE_NOT_SELECTED");
            return;
        }
        if (!watchWorker.connect(watchDevices.get(index).id)) {
            status.setText("WATCH_CONNECT_REJECTED");
            return;
        }
        refreshCombinedStatus();
    }

    private void disconnectWatch() {
        if (watchWorker != null) {
            watchWorker.disconnect();
        }
        refreshCombinedStatus();
    }

    private void listCameras() {
        try {
            String cameras = NativeBridge.nativeListCameras();
            cameraId = extractFirstCamera(cameras);
            status.setText(cameras);
            if (cameraId == null) {
                status.append("\nCAMERA_ID_UNAVAILABLE: no Camera2 NDK camera");
            } else {
                status.append("\nselected_camera_id=" + cameraId);
            }
        } catch (Throwable error) {
            status.setText("CAMERA_LIST_FAILED: " + error.getMessage());
        }
    }

    private void startCamera() {
        if (started) {
            return;
        }
        if (cameraId == null) {
            listCameras();
        }
        if (cameraId == null) {
            return;
        }
        try {
            if (nativeHandle == 0) {
                nativeHandle = NativeBridge.nativeCreate(cameraId, 640, 480, 30);
            }
            File cascade = ensureCascadeAsset();
            sessionOutputDirectory =
                    new File(
                            new File(getFilesDir(), "sessions"),
                            "session-" + System.currentTimeMillis());
            File model =
                    new File(new File(getFilesDir(), "models"), "efficientphys_pure.onnx");
            if (deepSelector.isChecked() && !model.isFile()) {
                status.setText(
                        "MODEL_LOAD_FAILED: import models/efficientphys_pure.onnx "
                                + "to app-private storage with adb run-as");
                return;
            }
            String configured =
                    NativeBridge.nativeConfigureProcessing(
                            nativeHandle,
                            methodSelector.getSelectedItem().toString(),
                            cascade.getAbsolutePath(),
                            sessionOutputDirectory.getAbsolutePath(),
                            deepSelector.isChecked(),
                            model.getAbsolutePath());
            if (!configured.startsWith("{")) {
                status.setText(configured);
                return;
            }
            alignmentHistory.clear();
            latestAlignment = null;
            lastAlignedWindowEndSec = null;
            sessionStartMonotonicSec = monotonicSec();
            String result = NativeBridge.nativeStart(nativeHandle);
            status.setText(result);
            if (!result.contains("\"state\":\"running\"")) {
                sessionStartMonotonicSec = null;
                return;
            }
            started = true;
            refreshCombinedStatus();
        } catch (Throwable error) {
            status.setText("CAMERA_START_FAILED: " + error.getMessage());
        }
    }

    private void stopCamera() {
        if (nativeHandle == 0) {
            started = false;
            return;
        }
        try {
            String stopped = NativeBridge.nativeStop(nativeHandle);
            exportWatchCsv();
            status.setText(stopped + "\n" + formatWatchStatusLine());
        } catch (Throwable error) {
            status.setText("CAMERA_STOP_FAILED: " + error.getMessage());
        } finally {
            started = false;
        }
    }

    private void exportWatchCsv() {
        if (sessionOutputDirectory == null || watchWorker == null) {
            return;
        }
        try {
            WatchContracts.WatchHeartRateSnapshot snapshot = watchWorker.snapshot(monotonicSec());
            String label =
                    snapshot.latestSample != null && !snapshot.latestSample.deviceName.isEmpty()
                            ? snapshot.latestSample.deviceName
                            : "HUAWEI WATCH GT 5 Pro";
            WatchCsvExport.writeSessionArtifacts(
                    sessionOutputDirectory,
                    snapshot.samples,
                    alignmentHistory,
                    sessionStartMonotonicSec,
                    label);
        } catch (Exception error) {
            status.append("\nWATCH_CSV_EXPORT_FAILED: " + error.getMessage());
        }
    }

    @Override
    protected void onStop() {
        stopCamera();
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        statusHandler.removeCallbacks(statusPoll);
        if (watchWorker != null) {
            watchWorker.close();
            watchWorker = null;
        }
        if (nativeHandle != 0) {
            NativeBridge.nativeDestroy(nativeHandle);
            nativeHandle = 0;
        }
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            int action = pendingAction;
            pendingAction = ACTION_NONE;
            if (grantResults.length == 0
                    || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                status.setText("CAMERA_PERMISSION_DENIED: permission was not granted");
                return;
            }
            performAction(action);
            return;
        }
        if (requestCode == BLE_PERMISSION_REQUEST) {
            int action = pendingBleAction;
            pendingBleAction = BLE_ACTION_NONE;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    status.setText("BLUETOOTH_PERMISSION_DENIED: permission was not granted");
                    return;
                }
            }
            performBleAction(action);
        }
    }

    private static String formatDouble(double value) {
        return String.format(Locale.US, "%.4g", value);
    }

    private static String extractFirstCamera(String json) {
        int array = json.indexOf("[\"");
        if (array < 0) {
            return null;
        }
        int start = array + 2;
        StringBuilder value = new StringBuilder();
        boolean escaped = false;
        for (int index = start; index < json.length(); index++) {
            char character = json.charAt(index);
            if (escaped) {
                value.append(character);
                escaped = false;
            } else if (character == '\\') {
                escaped = true;
            } else if (character == '"') {
                return value.toString();
            } else {
                value.append(character);
            }
        }
        return null;
    }

    private File ensureCascadeAsset() throws IOException {
        File cascade = new File(getFilesDir(), "haarcascade_frontalface_default.xml");
        if (cascade.isFile() && cascade.length() > 0) {
            return cascade;
        }
        File temporary = new File(cascade.getAbsolutePath() + ".tmp");
        try (InputStream input = getAssets().open("haarcascade_frontalface_default.xml");
                FileOutputStream output = new FileOutputStream(temporary)) {
            byte[] buffer = new byte[16 * 1024];
            int count;
            while ((count = input.read(buffer)) != -1) {
                output.write(buffer, 0, count);
            }
            output.getFD().sync();
        }
        if (!temporary.renameTo(cascade)) {
            temporary.delete();
            throw new IOException("could not install Haar cascade");
        }
        return cascade;
    }
}

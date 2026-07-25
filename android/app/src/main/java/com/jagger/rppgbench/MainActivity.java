package com.jagger.rppgbench;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;

import com.jagger.rppgbench.ui.HrMetricCard;
import com.jagger.rppgbench.ui.HrStatusFormatter;
import com.jagger.rppgbench.watch.AndroidBleBackend;
import com.jagger.rppgbench.watch.WatchAligner;
import com.jagger.rppgbench.watch.WatchBleWorker;
import com.jagger.rppgbench.watch.WatchContracts;
import com.jagger.rppgbench.watch.WatchCsvExport;

import org.json.JSONArray;
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

    private HrMetricCard traditionalCard;
    private HrMetricCard deepCard;
    private HrMetricCard watchCard;
    private TextView alignmentView;
    private TextView fpsLine;
    private ImageView roiImage;
    private TextView roiPlaceholder;
    private TextView diagnosticText;
    private LinearLayout cameraSection;
    private LinearLayout watchSection;
    private Spinner methodSelector;
    private Spinner cameraSelector;
    private Spinner watchDeviceSelector;
    private CheckBox deepSelector;
    private ArrayAdapter<String> cameraAdapter;
    private ArrayAdapter<String> watchDeviceAdapter;
    private final List<CameraEntry> cameraEntries = new ArrayList<>();
    private final List<WatchContracts.WatchDevice> watchDevices = new ArrayList<>();
    private final List<WatchContracts.WatchAlignmentResult> alignmentHistory = new ArrayList<>();

    private boolean cameraExpanded = true;
    private boolean watchExpanded = true;
    private boolean diagnosticExpanded = false;
    private String userMessage = "";
    private String lastCameraJson = "{}";

    private String cameraId;
    private long nativeHandle;
    private boolean started;
    private boolean cameraSpinnerInitializing;
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
        setContentView(R.layout.activity_main);

        traditionalCard = new HrMetricCard(findViewById(R.id.card_traditional));
        deepCard = new HrMetricCard(findViewById(R.id.card_deep));
        watchCard = new HrMetricCard(findViewById(R.id.card_watch));
        alignmentView = findViewById(R.id.alignment_line);
        fpsLine = findViewById(R.id.fps_line);
        roiImage = findViewById(R.id.roi_image);
        roiPlaceholder = findViewById(R.id.roi_placeholder);
        diagnosticText = findViewById(R.id.diagnostic_text);
        cameraSection = findViewById(R.id.camera_section);
        watchSection = findViewById(R.id.watch_section);

        methodSelector = findViewById(R.id.method_selector);
        ArrayAdapter<String> methodAdapter =
                new ArrayAdapter<>(
                        this,
                        android.R.layout.simple_spinner_item,
                        new String[] {"green", "pos", "chrom"});
        methodAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        methodSelector.setAdapter(methodAdapter);

        deepSelector = findViewById(R.id.deep_selector);

        cameraAdapter =
                new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, new ArrayList<>());
        cameraAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        cameraSelector = findViewById(R.id.camera_selector);
        cameraSelector.setAdapter(cameraAdapter);
        cameraSelector.setOnItemSelectedListener(
                new AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(
                            AdapterView<?> parent, View view, int position, long id) {
                        if (cameraSpinnerInitializing || position < 0
                                || position >= cameraEntries.size()) {
                            return;
                        }
                        String selectedId = cameraEntries.get(position).id;
                        if (selectedId.equals(cameraId)) {
                            return;
                        }
                        cameraId = selectedId;
                        if (started) {
                            restartCameraWithSelectedId();
                        }
                    }

                    @Override
                    public void onNothingSelected(AdapterView<?> parent) {}
                });

        watchDeviceAdapter =
                new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, new ArrayList<>());
        watchDeviceAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        watchDeviceSelector = findViewById(R.id.watch_device_selector);
        watchDeviceSelector.setAdapter(watchDeviceAdapter);

        findViewById(R.id.list_cameras)
                .setOnClickListener(view -> runWithCameraPermission(ACTION_LIST));
        findViewById(R.id.start_camera)
                .setOnClickListener(view -> runWithCameraPermission(ACTION_START));
        findViewById(R.id.stop_camera).setOnClickListener(view -> stopCamera());

        findViewById(R.id.scan_watch)
                .setOnClickListener(view -> runWithBlePermission(BLE_ACTION_SCAN));
        findViewById(R.id.connect_watch)
                .setOnClickListener(view -> runWithBlePermission(BLE_ACTION_CONNECT));
        findViewById(R.id.disconnect_watch).setOnClickListener(view -> disconnectWatch());

        setupFoldToggles();

        ensureWatchWorker();
        if (checkSelfPermission(Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            refreshCameraSpinner(false);
        }
        refreshCombinedStatus();
        statusHandler.post(statusPoll);
    }

    private void setupFoldToggles() {
        Button toggleCamera = findViewById(R.id.toggle_camera);
        Button toggleWatch = findViewById(R.id.toggle_watch);
        Button toggleDiagnostic = findViewById(R.id.toggle_diagnostic);

        cameraSection.setVisibility(cameraExpanded ? View.VISIBLE : View.GONE);
        watchSection.setVisibility(watchExpanded ? View.VISIBLE : View.GONE);
        diagnosticText.setVisibility(diagnosticExpanded ? View.VISIBLE : View.GONE);

        toggleCamera.setOnClickListener(
                view -> {
                    cameraExpanded = !cameraExpanded;
                    cameraSection.setVisibility(cameraExpanded ? View.VISIBLE : View.GONE);
                });
        toggleWatch.setOnClickListener(
                view -> {
                    watchExpanded = !watchExpanded;
                    watchSection.setVisibility(watchExpanded ? View.VISIBLE : View.GONE);
                });
        toggleDiagnostic.setOnClickListener(
                view -> {
                    diagnosticExpanded = !diagnosticExpanded;
                    diagnosticText.setVisibility(diagnosticExpanded ? View.VISIBLE : View.GONE);
                    if (diagnosticExpanded) {
                        refreshDiagnosticContent(lastCameraJson);
                    }
                });
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

    private void showUserMessage(String message) {
        userMessage = message == null ? "" : message;
        if (!userMessage.isEmpty()) {
            diagnosticExpanded = true;
            diagnosticText.setVisibility(View.VISIBLE);
        }
        refreshDiagnosticContent(lastCameraJson);
    }

    private void refreshCombinedStatus() {
        String cameraJson = "{}";
        if (started && nativeHandle != 0) {
            try {
                cameraJson = NativeBridge.nativeGetStatus(nativeHandle);
                maybeAlignFromCameraStatus(cameraJson);
                updateRoiImage();
            } catch (Throwable error) {
                showUserMessage("CAMERA_STATUS_FAILED: " + error.getMessage());
                refreshWatchDeviceSpinner();
                return;
            }
        }
        lastCameraJson = cameraJson;

        try {
            JSONObject json = new JSONObject(cameraJson);
            traditionalCard.bind("传统 rPPG", HrStatusFormatter.traditional(json));
            deepCard.bind("深度 EfficientPhys", HrStatusFormatter.deep(json));
        } catch (Exception error) {
            traditionalCard.bind("传统 rPPG", new HrStatusFormatter.CardView("--", "不可用"));
            deepCard.bind("深度 EfficientPhys", new HrStatusFormatter.CardView("--", "不可用"));
        }

        WatchContracts.WatchHeartRateSnapshot snapshot =
                watchWorker != null ? watchWorker.snapshot(monotonicSec()) : null;
        if (snapshot != null) {
            Integer bpm =
                    snapshot.latestSample != null ? snapshot.latestSample.bpm : null;
            watchCard.bind(
                    "广播心率",
                    HrStatusFormatter.watch(snapshot.status.name(), bpm, snapshot.errorCode));
        } else {
            watchCard.bind(
                    "广播心率", HrStatusFormatter.watch("DISCONNECTED", null, null));
        }

        if (latestAlignment != null) {
            alignmentView.setText(
                    HrStatusFormatter.alignmentLine(
                            latestAlignment.status.name(),
                            latestAlignment.absoluteErrorBpm,
                            latestAlignment.coverageRatio));
        } else {
            alignmentView.setText(HrStatusFormatter.alignmentLine(null, null, null));
        }

        updateFpsLine(cameraJson);

        if (diagnosticExpanded) {
            refreshDiagnosticContent(cameraJson);
        }
        refreshWatchDeviceSpinner();
    }

    private void updateFpsLine(String cameraJson) {
        if (!started || nativeHandle == 0) {
            fpsLine.setText(getString(R.string.fps_line_idle));
            return;
        }
        try {
            JSONObject json = new JSONObject(cameraJson);
            double measured = json.optDouble("measured_fps", 0.0);
            int targetMin = json.optInt("target_fps_min", 0);
            int targetMax = json.optInt("target_fps_max", 0);
            long dropped = json.optLong("dropped_frames", 0L);
            String measuredText =
                    measured > 0.0 ? formatDouble(measured) : "--";
            String targetText =
                    targetMin > 0 && targetMax > 0
                            ? targetMin + "–" + targetMax
                            : "--";
            fpsLine.setText(
                    String.format(
                            Locale.CHINA,
                            "采集 FPS: %s · 目标 %s · 丢帧 %d",
                            measuredText,
                            targetText,
                            dropped));
        } catch (Exception error) {
            fpsLine.setText(getString(R.string.fps_line_idle));
        }
    }

    private void refreshDiagnosticContent(String cameraJson) {
        StringBuilder builder = new StringBuilder();
        if (!userMessage.isEmpty()) {
            builder.append(userMessage).append('\n');
        }
        if (started && nativeHandle != 0) {
            builder.append(cameraJson);
        } else {
            try {
                builder.append(NativeBridge.nativeBuildIdentity());
            } catch (Throwable error) {
                builder.append("NATIVE_LOAD_FAILED: ").append(error.getClass().getSimpleName());
            }
        }
        builder.append('\n').append(formatWatchStatusLine());
        diagnosticText.setText(builder.toString());
    }

    private void updateRoiImage() {
        try {
            byte[] jpeg = NativeBridge.nativeGetRoiJpeg(nativeHandle);
            if (jpeg != null && jpeg.length > 0) {
                Bitmap bitmap = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
                if (bitmap != null) {
                    roiImage.setImageBitmap(bitmap);
                    roiImage.setVisibility(View.VISIBLE);
                    roiPlaceholder.setVisibility(View.GONE);
                    return;
                }
            }
        } catch (Throwable ignored) {
            // Keep cards visible even if ROI decode fails.
        }
        roiImage.setVisibility(View.GONE);
        roiPlaceholder.setVisibility(View.VISIBLE);
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
            showUserMessage("WATCH_SCAN_BUSY");
            return;
        }
        userMessage = "";
        refreshCombinedStatus();
    }

    private void connectWatch() {
        ensureWatchWorker();
        int index = watchDeviceSelector.getSelectedItemPosition();
        if (index < 0 || index >= watchDevices.size()) {
            showUserMessage("WATCH_DEVICE_NOT_SELECTED");
            return;
        }
        if (!watchWorker.connect(watchDevices.get(index).id)) {
            showUserMessage("WATCH_CONNECT_REJECTED");
            return;
        }
        userMessage = "";
        refreshCombinedStatus();
    }

    private void disconnectWatch() {
        if (watchWorker != null) {
            watchWorker.disconnect();
        }
        userMessage = "";
        refreshCombinedStatus();
    }

    private void listCameras() {
        try {
            String cameras = NativeBridge.nativeListCameras();
            refreshCameraSpinner(true);
            showUserMessage(cameras);
        } catch (Throwable error) {
            showUserMessage("CAMERA_LIST_FAILED: " + error.getMessage());
        }
    }

    private void refreshCameraSpinner(boolean showSelectionMessage) {
        try {
            String camerasJson = NativeBridge.nativeListCameras();
            populateCameraSpinner(camerasJson);
            if (showSelectionMessage && cameraId != null) {
                showUserMessage(camerasJson + "\nselected_camera_id=" + cameraId);
            }
        } catch (Throwable error) {
            showUserMessage("CAMERA_LIST_FAILED: " + error.getMessage());
        }
    }

    private void populateCameraSpinner(String camerasJson) {
        List<CameraEntry> parsed = parseCameraEntries(camerasJson);
        if (sameCameras(cameraEntries, parsed)) {
            return;
        }
        cameraEntries.clear();
        cameraEntries.addAll(parsed);
        cameraAdapter.clear();
        int defaultIndex = 0;
        for (int index = 0; index < cameraEntries.size(); index++) {
            CameraEntry entry = cameraEntries.get(index);
            cameraAdapter.add(formatCameraLabel(entry));
            if ("front".equals(entry.facing)) {
                defaultIndex = index;
            }
        }
        cameraAdapter.notifyDataSetChanged();
        if (cameraEntries.isEmpty()) {
            cameraId = null;
            return;
        }
        cameraSpinnerInitializing = true;
        cameraSelector.setSelection(defaultIndex, false);
        cameraSpinnerInitializing = false;
        cameraId = cameraEntries.get(defaultIndex).id;
    }

    private static List<CameraEntry> parseCameraEntries(String camerasJson) {
        List<CameraEntry> entries = new ArrayList<>();
        try {
            JSONArray cameras = new JSONObject(camerasJson).getJSONArray("cameras");
            for (int index = 0; index < cameras.length(); index++) {
                JSONObject camera = cameras.getJSONObject(index);
                entries.add(
                        new CameraEntry(
                                camera.getString("id"),
                                camera.optString("facing", "unknown")));
            }
        } catch (Exception ignored) {
            // Leave entries empty; caller shows failure elsewhere.
        }
        return entries;
    }

    private static boolean sameCameras(List<CameraEntry> left, List<CameraEntry> right) {
        if (left.size() != right.size()) {
            return false;
        }
        for (int index = 0; index < left.size(); index++) {
            CameraEntry a = left.get(index);
            CameraEntry b = right.get(index);
            if (!a.id.equals(b.id) || !a.facing.equals(b.facing)) {
                return false;
            }
        }
        return true;
    }

    private static String formatCameraLabel(CameraEntry entry) {
        String facingLabel;
        switch (entry.facing) {
            case "front":
                facingLabel = "前置";
                break;
            case "back":
                facingLabel = "后置";
                break;
            case "external":
                facingLabel = "外置";
                break;
            default:
                facingLabel = "未知";
                break;
        }
        return facingLabel + " (" + entry.id + ")";
    }

    private String selectedCameraId() {
        int index = cameraSelector.getSelectedItemPosition();
        if (index >= 0 && index < cameraEntries.size()) {
            return cameraEntries.get(index).id;
        }
        return cameraId;
    }

    private void restartCameraWithSelectedId() {
        stopCameraSilently();
        startCamera();
    }

    private void stopCameraSilently() {
        if (nativeHandle == 0) {
            started = false;
            return;
        }
        try {
            NativeBridge.nativeStop(nativeHandle);
            exportWatchCsv();
        } catch (Throwable ignored) {
            // Best effort before switching cameras.
        } finally {
            started = false;
            NativeBridge.nativeDestroy(nativeHandle);
            nativeHandle = 0;
            roiImage.setVisibility(View.GONE);
            roiPlaceholder.setVisibility(View.VISIBLE);
        }
    }

    private void startCamera() {
        if (started) {
            return;
        }
        if (cameraEntries.isEmpty()) {
            refreshCameraSpinner(false);
        }
        cameraId = selectedCameraId();
        if (cameraId == null) {
            showUserMessage("CAMERA_ID_UNAVAILABLE: no Camera2 NDK camera");
            return;
        }
        try {
            if (nativeHandle != 0) {
                NativeBridge.nativeDestroy(nativeHandle);
                nativeHandle = 0;
            }
            nativeHandle = NativeBridge.nativeCreate(cameraId, 640, 480, 30);
            File cascade = ensureCascadeAsset();
            sessionOutputDirectory =
                    new File(
                            new File(getFilesDir(), "sessions"),
                            "session-" + System.currentTimeMillis());
            File model =
                    new File(new File(getFilesDir(), "models"), "efficientphys_pure.onnx");
            if (deepSelector.isChecked() && !model.isFile()) {
                showUserMessage(
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
                showUserMessage(configured);
                return;
            }
            alignmentHistory.clear();
            latestAlignment = null;
            lastAlignedWindowEndSec = null;
            sessionStartMonotonicSec = monotonicSec();
            String result = NativeBridge.nativeStart(nativeHandle);
            if (!result.contains("\"state\":\"running\"")) {
                showUserMessage(result);
                sessionStartMonotonicSec = null;
                return;
            }
            started = true;
            userMessage = "";
            refreshCombinedStatus();
        } catch (Throwable error) {
            showUserMessage("CAMERA_START_FAILED: " + error.getMessage());
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
            showUserMessage(stopped + "\n" + formatWatchStatusLine());
        } catch (Throwable error) {
            showUserMessage("CAMERA_STOP_FAILED: " + error.getMessage());
        } finally {
            started = false;
            roiImage.setVisibility(View.GONE);
            roiPlaceholder.setVisibility(View.VISIBLE);
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
            showUserMessage("WATCH_CSV_EXPORT_FAILED: " + error.getMessage());
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
                showUserMessage("CAMERA_PERMISSION_DENIED: permission was not granted");
                return;
            }
            performAction(action);
            if (action == ACTION_NONE
                    && checkSelfPermission(Manifest.permission.CAMERA)
                            == PackageManager.PERMISSION_GRANTED) {
                refreshCameraSpinner(false);
            }
            return;
        }
        if (requestCode == BLE_PERMISSION_REQUEST) {
            int action = pendingBleAction;
            pendingBleAction = BLE_ACTION_NONE;
            for (int result : grantResults) {
                if (result != PackageManager.PERMISSION_GRANTED) {
                    showUserMessage("BLUETOOTH_PERMISSION_DENIED: permission was not granted");
                    return;
                }
            }
            performBleAction(action);
        }
    }

    private static final class CameraEntry {
        private final String id;
        private final String facing;

        private CameraEntry(String id, String facing) {
            this.id = id;
            this.facing = facing;
        }
    }

    private static String formatDouble(double value) {
        return String.format(Locale.US, "%.4g", value);
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

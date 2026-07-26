package com.jagger.rppgbench;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;

import com.jagger.rppgbench.ui.FaceBoxOverlay;
import com.jagger.rppgbench.ui.HrMetricCard;
import com.jagger.rppgbench.ui.HrStatusFormatter;
import com.jagger.rppgbench.ui.PpgWaveformCard;
import com.jagger.rppgbench.ui.PpgWaveformSnapshot;
import com.jagger.rppgbench.ui.PpgWaveformState;
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
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

public final class MainActivity extends Activity {
    private static final int CAMERA_CAPTURE_WIDTH = 640;
    private static final int CAMERA_CAPTURE_HEIGHT = 480;
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
    private PpgWaveformCard traditionalWaveformCard;
    private PpgWaveformCard deepWaveformCard;
    private PpgWaveformSnapshot traditionalWaveform;
    private PpgWaveformSnapshot deepWaveform;
    private long traditionalWaveformRevision;
    private long deepWaveformRevision;
    private TextView alignmentView;
    private TextView fpsLine;
    private FrameLayout previewContainer;
    private TextureView previewSurface;
    private FaceBoxOverlay faceBoxOverlay;
    private TextView previewPlaceholder;
    private ImageView roiImage;
    private TextView roiPlaceholder;
    private TextView diagnosticText;
    private LinearLayout cameraSection;
    private LinearLayout watchSection;
    private Spinner methodSelector;
    private Spinner cameraSelector;
    private Spinner watchDeviceSelector;
    private Spinner deepSelector;
    private Button listCamerasButton;
    private Button startCameraButton;
    private Button stopCameraButton;
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
    private DeepModelSelection activeDeepSelection =
            DeepModelSelection.fromSpinnerPosition(0);

    private String cameraId;
    private long nativeHandle;
    private boolean started;
    private boolean previewSurfaceReady;
    private boolean pendingPreviewBinding;
    private boolean pendingCameraStart;
    private boolean lifecycleStopped;
    private boolean startWorkerSubmitted;
    private CameraStartRequest pendingStartRequest;
    private long pendingStartGeneration;
    private final CameraStartGeneration cameraStartGeneration = new CameraStartGeneration();
    private final CameraUiSessionPolicy cameraUiSessionPolicy =
            new CameraUiSessionPolicy();
    private final ExecutorService cameraStartExecutor =
            Executors.newSingleThreadExecutor(
                    runnable -> {
                        Thread thread = new Thread(runnable, "rppg-camera-start");
                        thread.setDaemon(true);
                        return thread;
                    });
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
        traditionalWaveformCard = new PpgWaveformCard(findViewById(R.id.waveform_traditional));
        deepWaveformCard = new PpgWaveformCard(findViewById(R.id.waveform_deep));
        initializeWaveformCards();
        alignmentView = findViewById(R.id.alignment_line);
        fpsLine = findViewById(R.id.fps_line);
        previewContainer = findViewById(R.id.preview_container);
        previewSurface = findViewById(R.id.preview_surface);
        faceBoxOverlay = findViewById(R.id.face_box_overlay);
        previewPlaceholder = findViewById(R.id.preview_placeholder);
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
        ArrayAdapter<String> deepModelAdapter =
                new ArrayAdapter<>(
                        this,
                        android.R.layout.simple_spinner_item,
                        DeepModelSelection.spinnerLabels());
        deepModelAdapter.setDropDownViewResource(
                android.R.layout.simple_spinner_dropdown_item);
        deepSelector.setAdapter(deepModelAdapter);
        deepSelector.setSelection(0);

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
                        updatePreviewMirror();
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

        listCamerasButton = findViewById(R.id.list_cameras);
        startCameraButton = findViewById(R.id.start_camera);
        stopCameraButton = findViewById(R.id.stop_camera);
        listCamerasButton.setOnClickListener(view -> runWithCameraPermission(ACTION_LIST));
        startCameraButton.setOnClickListener(view -> runWithCameraPermission(ACTION_START));
        stopCameraButton.setOnClickListener(view -> stopCamera());
        applyCameraUiDecision(cameraUiSessionPolicy.current());

        findViewById(R.id.scan_watch)
                .setOnClickListener(view -> runWithBlePermission(BLE_ACTION_SCAN));
        findViewById(R.id.connect_watch)
                .setOnClickListener(view -> runWithBlePermission(BLE_ACTION_CONNECT));
        findViewById(R.id.disconnect_watch).setOnClickListener(view -> disconnectWatch());

        setupPreviewSurface();
        setupFoldToggles();

        ensureWatchWorker();
        if (checkSelfPermission(Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            refreshCameraSpinner(false);
        }
        refreshCombinedStatus();
        statusHandler.post(statusPoll);
    }

    private void setupPreviewSurface() {
        previewSurface.setSurfaceTextureListener(
                new TextureView.SurfaceTextureListener() {
                    @Override
                    public void onSurfaceTextureAvailable(
                            SurfaceTexture surfaceTexture, int width, int height) {
                        surfaceTexture.setDefaultBufferSize(
                                CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT);
                        previewSurfaceReady = true;
                        applyPreviewAspect();
                        // TextureView only gets a Surface after the container is visible.
                        // Bind + start must wait for this callback; binding while running is rejected.
                        if (pendingCameraStart) {
                            finishStartCamera();
                            return;
                        }
                        if (pendingPreviewBinding && !started) {
                            bindPreviewSurface(surfaceTexture);
                        }
                    }

                    @Override
                    public void onSurfaceTextureSizeChanged(
                            SurfaceTexture surfaceTexture, int width, int height) {
                        applyPreviewAspect();
                    }

                    @Override
                    public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
                        previewSurfaceReady = false;
                        pendingPreviewBinding = false;
                        if (!started && !pendingCameraStart) {
                            releasePreviewSurface();
                        }
                        return true;
                    }

                    @Override
                    public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {}
                });
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
        String cameraJson = lastCameraJson;
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
            deepCard.bind(deepCardTitle(activeDeepSelection), HrStatusFormatter.deep(json));
            if (started && nativeHandle != 0) {
                updateWaveformCards(json);
            }
        } catch (Exception error) {
            traditionalCard.bind("传统 rPPG", new HrStatusFormatter.CardView("--", "不可用"));
            deepCard.bind(
                    deepCardTitle(activeDeepSelection),
                    new HrStatusFormatter.CardView("--", "不可用"));
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
        updateFaceBoxOverlay(cameraJson);
        updatePreviewState(cameraJson);

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

    private void updateWaveformCards(JSONObject status) {
        long traditionalRevision = status.optLong("traditional_waveform_revision", 0);
        long deepRevision = status.optLong("deep_waveform_revision", 0);
        if (traditionalRevision > 0 && traditionalRevision != traditionalWaveformRevision) {
            PpgWaveformSnapshot loaded = loadWaveform(false);
            if (loaded != null) {
                traditionalWaveform = loaded;
                traditionalWaveformRevision = loaded.revision;
            }
        }
        if (deepRevision > 0 && deepRevision != deepWaveformRevision) {
            PpgWaveformSnapshot loaded = loadWaveform(true);
            if (loaded != null) {
                deepWaveform = loaded;
                deepWaveformRevision = loaded.revision;
            }
        }

        String traditionalState = getString(R.string.waveform_traditional_sampling);
        if (traditionalWaveform != null) {
            traditionalState = traditionalWaveform.valid
                    ? traditionalWaveform.method + " · " + traditionalWaveform.sampleCount()
                            + " 点 · " + String.format(Locale.US, "%.1f 秒",
                                    -traditionalWaveform.relativeStartSeconds())
                    : "信号质量低：" + traditionalWaveform.invalidReason;
        }
        traditionalWaveformCard.bind(getString(R.string.waveform_traditional_title),
                traditionalState, traditionalWaveform);

        boolean deepEnabled = status.optBoolean("deep_enabled", false);
        int collected = status.optInt("deep_frames_collected", 0);
        int required = status.optInt("deep_frames_required", 180);
        boolean available = status.optBoolean("deep_result_available", false);
        boolean waveformValid = status.optBoolean("deep_result_valid", false);
        boolean stabilityValid = status.optBoolean("deep_stability_valid", false);
        String reason = waveformValid
                ? status.optString("deep_correction_reason", "")
                : status.optString("deep_invalid_reason", "");
        String deepState = PpgWaveformState.deep(
                activeDeepSelection.modelLabel,
                deepEnabled,
                collected,
                required,
                available,
                waveformValid && stabilityValid,
                reason);
        deepWaveformCard.bind(
                deepWaveformTitle(activeDeepSelection), deepState, deepWaveform);
    }

    private PpgWaveformSnapshot loadWaveform(boolean deep) {
        try {
            JSONObject metadata = new JSONObject(
                    NativeBridge.nativeGetWaveformMetadata(nativeHandle, deep));
            if (!metadata.optBoolean("available", false)) {
                return null;
            }
            float[] values = NativeBridge.nativeGetWaveformValues(nativeHandle, deep);
            int expected = metadata.optInt("sample_count", 0);
            if (values == null || values.length != expected) {
                return null;
            }
            return new PpgWaveformSnapshot(
                    metadata.optLong("revision", 0), metadata.optString("method", ""),
                    metadata.optDouble("sample_rate_hz", 30.0),
                    metadata.optBoolean("is_valid", false),
                    metadata.optString("invalid_reason", ""), values);
        } catch (Throwable error) {
            return null;
        }
    }

    private void clearWaveformsForNewSession() {
        traditionalWaveform = null;
        deepWaveform = null;
        traditionalWaveformRevision = 0;
        deepWaveformRevision = 0;
        initializeWaveformCards();
    }

    private void initializeWaveformCards() {
        if (traditionalWaveformCard != null) {
            traditionalWaveformCard.clear(getString(R.string.waveform_traditional_title),
                    getString(R.string.waveform_traditional_sampling));
        }
        if (deepWaveformCard != null) {
            deepWaveformCard.clear(deepWaveformTitle(activeDeepSelection),
                    getString(R.string.waveform_deep_disabled));
        }
    }

    private void updateFaceBoxOverlay(String cameraJson) {
        if (faceBoxOverlay == null) {
            return;
        }
        if (!started || nativeHandle == 0) {
            faceBoxOverlay.clearFaceRect();
            return;
        }
        try {
            JSONObject json = new JSONObject(cameraJson);
            JSONObject faceRect = json.optJSONObject("face_rect");
            if (faceRect != null) {
                faceBoxOverlay.setFaceRect(
                        (float) faceRect.getDouble("x"),
                        (float) faceRect.getDouble("y"),
                        (float) faceRect.getDouble("w"),
                        (float) faceRect.getDouble("h"));
            } else {
                faceBoxOverlay.clearFaceRect();
            }
        } catch (Exception error) {
            faceBoxOverlay.clearFaceRect();
        }
    }

    private void updatePreviewState(String cameraJson) {
        if (previewContainer == null) {
            return;
        }
        if (pendingCameraStart) {
            showPreviewContainerForBinding();
            return;
        }
        if (!started || nativeHandle == 0) {
            previewContainer.setVisibility(View.GONE);
            previewPlaceholder.setVisibility(View.GONE);
            previewSurface.setVisibility(View.VISIBLE);
            return;
        }
        previewContainer.setVisibility(View.VISIBLE);
        boolean previewEnabled = false;
        try {
            previewEnabled = new JSONObject(cameraJson).optBoolean("preview_enabled", false);
        } catch (Exception ignored) {
            // Keep preview visible; placeholder covers unavailable preview.
        }
        previewSurface.setVisibility(previewEnabled ? View.VISIBLE : View.GONE);
        previewPlaceholder.setVisibility(previewEnabled ? View.GONE : View.VISIBLE);
        if (!previewEnabled) {
            previewPlaceholder.setText(R.string.preview_placeholder);
        }
    }

    private void showPreviewContainerForBinding() {
        previewContainer.setVisibility(View.VISIBLE);
        previewSurface.setVisibility(View.VISIBLE);
        previewPlaceholder.setVisibility(View.GONE);
        applyPreviewAspect();
        updatePreviewMirror();
    }

    /**
     * Sensor buffer is landscape 640×480. Phone is portrait: rotate preview to
     * match system camera, and size the container to the upright 3:4 aspect.
     */
    private void applyPreviewAspect() {
        if (previewContainer == null || previewSurface == null) {
            return;
        }
        previewContainer.post(
                () -> {
                    int width = previewContainer.getWidth();
                    if (width <= 0) {
                        return;
                    }
                    int relative = computeRelativeRotationDegrees();
                    boolean swap = relative == 90 || relative == 270;
                    int aspectWidth = swap ? CAMERA_CAPTURE_HEIGHT : CAMERA_CAPTURE_WIDTH;
                    int aspectHeight = swap ? CAMERA_CAPTURE_WIDTH : CAMERA_CAPTURE_HEIGHT;
                    int height =
                            Math.round(width * (aspectHeight / (float) aspectWidth));
                    ViewGroup.LayoutParams layoutParams = previewContainer.getLayoutParams();
                    if (layoutParams != null && layoutParams.height != height) {
                        layoutParams.height = height;
                        previewContainer.setLayoutParams(layoutParams);
                    }
                    previewSurface.post(this::configurePreviewTransform);
                });
    }

    private void configurePreviewTransform() {
        if (previewSurface == null) {
            return;
        }
        int viewWidth = previewSurface.getWidth();
        int viewHeight = previewSurface.getHeight();
        if (viewWidth <= 0 || viewHeight <= 0) {
            return;
        }
        int relative = computeRelativeRotationDegrees();
        Matrix matrix = new Matrix();
        float centerX = viewWidth / 2f;
        float centerY = viewHeight / 2f;

        boolean swap = relative == 90 || relative == 270;
        float bufferAspect =
                swap
                        ? (CAMERA_CAPTURE_HEIGHT / (float) CAMERA_CAPTURE_WIDTH)
                        : (CAMERA_CAPTURE_WIDTH / (float) CAMERA_CAPTURE_HEIGHT);
        float viewAspect = viewWidth / (float) viewHeight;
        float scaleX = 1f;
        float scaleY = 1f;
        if (bufferAspect > viewAspect) {
            scaleX = bufferAspect / viewAspect;
        } else {
            scaleY = viewAspect / bufferAspect;
        }

        matrix.postScale(scaleX, scaleY, centerX, centerY);
        matrix.postRotate(
                PreviewRotation.textureTransformDegrees(relative), centerX, centerY);
        previewSurface.setTransform(matrix);
        updatePreviewMirror();
    }

    private int computeRelativeRotationDegrees() {
        int sensor = selectedSensorOrientation();
        int display = displayRotationDegrees();
        if (isSelectedCameraFront()) {
            return ((sensor + display) % 360 + 360) % 360;
        }
        return ((sensor - display) % 360 + 360) % 360;
    }

    private int displayRotationDegrees() {
        int rotation = getWindowManager().getDefaultDisplay().getRotation();
        switch (rotation) {
            case Surface.ROTATION_90:
                return 90;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_0:
            default:
                return 0;
        }
    }

    private int selectedSensorOrientation() {
        String selectedId = selectedCameraId();
        if (selectedId == null) {
            return 0;
        }
        for (CameraEntry entry : cameraEntries) {
            if (selectedId.equals(entry.id)) {
                return entry.sensorOrientation;
            }
        }
        return 0;
    }

    private void updatePreviewMirror() {
        if (previewSurface == null || faceBoxOverlay == null) {
            return;
        }
        boolean mirror = isSelectedCameraFront();
        previewSurface.setScaleX(mirror ? -1f : 1f);
        faceBoxOverlay.setMirrorHorizontal(mirror);
    }

    private boolean isSelectedCameraFront() {
        String selectedId = selectedCameraId();
        if (selectedId == null) {
            return false;
        }
        for (CameraEntry entry : cameraEntries) {
            if (selectedId.equals(entry.id)) {
                return "front".equals(entry.facing);
            }
        }
        return false;
    }

    private void bindPreviewSurface(SurfaceTexture surfaceTexture) {
        if (nativeHandle == 0 || surfaceTexture == null) {
            return;
        }
        pendingPreviewBinding = false;
        try {
            NativeBridge.nativeSetPreviewSurface(nativeHandle, new Surface(surfaceTexture));
        } catch (Throwable error) {
            showUserMessage("PREVIEW_SURFACE_FAILED: " + error.getMessage());
        }
    }

    private void releasePreviewSurface() {
        if (nativeHandle == 0) {
            return;
        }
        try {
            NativeBridge.nativeSetPreviewSurface(nativeHandle, null);
        } catch (Throwable ignored) {
            // Best effort when TextureView is torn down.
        }
    }

    private void preparePreviewSurfaceBinding() {
        pendingPreviewBinding = true;
        showPreviewContainerForBinding();
        if (previewSurfaceReady && previewSurface.isAvailable()) {
            SurfaceTexture surfaceTexture = previewSurface.getSurfaceTexture();
            if (surfaceTexture != null) {
                surfaceTexture.setDefaultBufferSize(
                        CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT);
            }
        }
    }

    private void setCameraControlsLocked(boolean locked) {
        methodSelector.setEnabled(!locked);
        deepSelector.setEnabled(!locked);
        cameraSelector.setEnabled(!locked);
        listCamerasButton.setEnabled(!locked);
        startCameraButton.setEnabled(!locked);
        stopCameraButton.setEnabled(locked);
    }

    private void applyCameraUiDecision(CameraUiSessionPolicy.Decision decision) {
        if (decision.clearHistory) {
            clearWaveformsForNewSession();
        }
        setCameraControlsLocked(decision.selectorsLocked);
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
                                camera.optString("facing", "unknown"),
                                camera.optInt("sensor_orientation", 0)));
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
            if (!a.id.equals(b.id)
                    || !a.facing.equals(b.facing)
                    || a.sensorOrientation != b.sensorOrientation) {
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
        cameraStartGeneration.cancel();
        CameraUiSessionPolicy.Decision uiDecision =
                cameraUiSessionPolicy.stopRequested();
        pendingCameraStart = false;
        startWorkerSubmitted = false;
        pendingStartRequest = null;
        statusHandler.removeCallbacks(finishStartWhenPreviewReady);
        pendingPreviewBinding = false;
        if (nativeHandle == 0) {
            started = false;
            applyCameraUiDecision(uiDecision);
            return;
        }
        try {
            String stopped = NativeBridge.nativeStop(nativeHandle);
            if (uiDecision.requestFinalSnapshot && uiDecision.retainLastResult) {
                retainLatestCameraResult(stopped);
            }
            exportWatchCsv();
        } catch (Throwable ignored) {
            // Best effort before switching cameras.
        } finally {
            started = false;
            applyCameraUiDecision(uiDecision);
            NativeBridge.nativeDestroy(nativeHandle);
            nativeHandle = 0;
            roiImage.setVisibility(View.GONE);
            roiPlaceholder.setVisibility(View.VISIBLE);
            previewContainer.setVisibility(View.GONE);
            faceBoxOverlay.clearFaceRect();
        }
    }

    private final Runnable finishStartWhenPreviewReady =
            () -> {
                if (pendingCameraStart && !started) {
                    finishStartCamera();
                }
            };

    private void startCamera() {
        if (lifecycleStopped || started || pendingCameraStart) {
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
        if (nativeHandle != 0) {
            NativeBridge.nativeDestroy(nativeHandle);
            nativeHandle = 0;
        }
        File filesDirectory = getFilesDir();
        DeepModelSelection deepSelection = DeepModelSelection.fromSpinnerPosition(
                deepSelector.getSelectedItemPosition());
        pendingStartRequest =
                new CameraStartRequest(
                        cameraId,
                        methodSelector.getSelectedItem().toString(),
                        deepSelection,
                        ModelIntegrity.modelFile(
                                new File(filesDirectory, "models"), deepSelection),
                        filesDirectory,
                        getAssets(),
                        new File(
                                new File(filesDirectory, "sessions"),
                                "session-" + System.currentTimeMillis()),
                        displayRotationDegrees());
        pendingStartGeneration = cameraStartGeneration.begin();
        pendingCameraStart = true;
        startWorkerSubmitted = false;
        applyCameraUiDecision(cameraUiSessionPolicy.begin());
        showUserMessage("正在后台校验模型并启动相机…");
        preparePreviewSurfaceBinding();
        if (previewSurfaceReady && previewSurface.isAvailable()) {
            finishStartCamera();
        } else {
            statusHandler.postDelayed(finishStartWhenPreviewReady, 750);
        }
    }

    private void finishStartCamera() {
        if (!pendingCameraStart || started || startWorkerSubmitted
                || pendingStartRequest == null) {
            return;
        }
        statusHandler.removeCallbacks(finishStartWhenPreviewReady);
        pendingPreviewBinding = false;
        Surface surface = null;
        if (previewSurfaceReady && previewSurface.isAvailable()) {
            SurfaceTexture surfaceTexture = previewSurface.getSurfaceTexture();
            if (surfaceTexture != null) {
                surfaceTexture.setDefaultBufferSize(
                        CAMERA_CAPTURE_WIDTH, CAMERA_CAPTURE_HEIGHT);
                surface = new Surface(surfaceTexture);
            }
        }
        startWorkerSubmitted = true;
        CameraStartRequest request = pendingStartRequest;
        long generation = pendingStartGeneration;
        Surface capturedSurface = surface;
        try {
            cameraStartExecutor.execute(
                    () -> runCameraStart(generation, request, capturedSurface));
        } catch (RejectedExecutionException error) {
            if (capturedSurface != null) {
                capturedSurface.release();
            }
            if (cameraStartGeneration.isCurrent(generation)) {
                pendingCameraStart = false;
                startWorkerSubmitted = false;
                pendingStartRequest = null;
                applyCameraUiDecision(cameraUiSessionPolicy.fail());
                showUserMessage("CAMERA_START_FAILED: background executor is unavailable");
            }
        }
    }

    private void runCameraStart(
            long generation, CameraStartRequest request, Surface preview) {
        long handle = 0;
        boolean transferredToActivity = false;
        try (PreparedModel prepared = PreparedModel.prepare(
                request.deepSelection, request.modelFile)) {
            if (!cameraStartGeneration.isCurrent(generation)) {
                return;
            }
            File cascade = ensureCascadeAsset(request.filesDirectory, request.assets);
            handle =
                    NativeBridge.nativeCreate(
                            request.cameraId,
                            CAMERA_CAPTURE_WIDTH,
                            CAMERA_CAPTURE_HEIGHT,
                            30);
            String configured =
                    NativeBridge.nativeConfigureProcessing(
                            handle,
                            request.method,
                            cascade.getAbsolutePath(),
                            request.sessionDirectory.getAbsolutePath(),
                            request.deepSelection.canonicalName,
                            prepared.nativePath());
            if (!configured.startsWith("{")) {
                throw new IllegalStateException(configured);
            }
            NativeBridge.nativeSetDisplayRotation(handle, request.displayRotationDegrees);
            if (preview != null) {
                NativeBridge.nativeSetPreviewSurface(handle, preview);
            }
            if (!cameraStartGeneration.isCurrent(generation)) {
                return;
            }
            double startMonotonicSec = monotonicSec();
            String result = NativeBridge.nativeStart(handle);
            if (!result.contains("\"state\":\"running\"")) {
                throw new IllegalStateException(result);
            }

            AtomicBoolean deliveryOpen = new AtomicBoolean(true);
            AtomicBoolean accepted = new AtomicBoolean(false);
            CountDownLatch delivered = new CountDownLatch(1);
            long completedHandle = handle;
            boolean posted =
                    statusHandler.post(
                            () -> {
                                try {
                                    if (deliveryOpen.compareAndSet(true, false)
                                            && cameraStartGeneration.isCurrent(generation)) {
                                        // Transfer ownership before UI work can throw.
                                        accepted.set(true);
                                        acceptStartedCamera(
                                                request,
                                                completedHandle,
                                                result,
                                                startMonotonicSec);
                                    }
                                } finally {
                                    delivered.countDown();
                                }
                            });
            if (!posted) {
                deliveryOpen.set(false);
                delivered.countDown();
            }
            boolean interrupted = false;
            while (true) {
                try {
                    delivered.await();
                    break;
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
            transferredToActivity = accepted.get();
        } catch (Throwable error) {
            if (!transferredToActivity) {
                postCameraStartFailure(generation, error.getMessage());
            }
        } finally {
            if (preview != null) {
                preview.release();
            }
            if (handle != 0 && !transferredToActivity) {
                stopAndDestroyNative(handle);
            }
        }
    }

    private void acceptStartedCamera(
            CameraStartRequest request, long handle, String result, double startMonotonicSec) {
        nativeHandle = handle;
        activeDeepSelection = request.deepSelection;
        sessionOutputDirectory = request.sessionDirectory;
        sessionStartMonotonicSec = startMonotonicSec;
        pendingCameraStart = false;
        startWorkerSubmitted = false;
        pendingStartRequest = null;
        started = true;
        alignmentHistory.clear();
        latestAlignment = null;
        lastAlignedWindowEndSec = null;
        applyCameraUiDecision(cameraUiSessionPolicy.accept());
        applyPreviewAspect();
        userMessage = "";
        refreshCombinedStatus();
        try {
            if (!new JSONObject(result).optBoolean("preview_enabled", false)) {
                showUserMessage(
                        "预览未接入（仅分析通路）。请看诊断里 preview_enabled / "
                                + "preview_unavailable。");
            }
        } catch (Exception ignored) {
            // Status poll will refresh preview_enabled.
        }
    }

    private void postCameraStartFailure(long generation, String message) {
        statusHandler.post(
                () -> {
                    if (!cameraStartGeneration.isCurrent(generation)) {
                        return;
                    }
                    pendingCameraStart = false;
                    startWorkerSubmitted = false;
                    pendingStartRequest = null;
                    pendingPreviewBinding = false;
                    applyCameraUiDecision(cameraUiSessionPolicy.fail());
                    previewContainer.setVisibility(View.GONE);
                    showUserMessage("CAMERA_START_FAILED: " + message);
                });
    }

    private static void stopAndDestroyNative(long handle) {
        try {
            NativeBridge.nativeStop(handle);
        } catch (Throwable ignored) {
            // The handle may have failed before reaching the running state.
        }
        try {
            NativeBridge.nativeDestroy(handle);
        } catch (Throwable ignored) {
            // Best-effort cleanup for canceled/failed background starts.
        }
    }

    private void stopCamera() {
        cameraStartGeneration.cancel();
        CameraUiSessionPolicy.Decision uiDecision =
                cameraUiSessionPolicy.stopRequested();
        pendingCameraStart = false;
        startWorkerSubmitted = false;
        pendingStartRequest = null;
        statusHandler.removeCallbacks(finishStartWhenPreviewReady);
        pendingPreviewBinding = false;
        if (nativeHandle == 0) {
            started = false;
            applyCameraUiDecision(uiDecision);
            previewContainer.setVisibility(View.GONE);
            return;
        }
        try {
            String stopped = NativeBridge.nativeStop(nativeHandle);
            if (uiDecision.requestFinalSnapshot && uiDecision.retainLastResult) {
                retainLatestCameraResult(stopped);
            }
            exportWatchCsv();
            showUserMessage(stopped + "\n" + formatWatchStatusLine());
        } catch (Throwable error) {
            showUserMessage("CAMERA_STOP_FAILED: " + error.getMessage());
        } finally {
            started = false;
            applyCameraUiDecision(uiDecision);
            releasePreviewSurface();
            roiImage.setVisibility(View.GONE);
            roiPlaceholder.setVisibility(View.VISIBLE);
            previewContainer.setVisibility(View.GONE);
            faceBoxOverlay.clearFaceRect();
        }
    }

    private void retainLatestCameraResult(String statusJson) {
        try {
            JSONObject status = new JSONObject(statusJson);
            lastCameraJson = statusJson;
            traditionalCard.bind("传统 rPPG", HrStatusFormatter.traditional(status));
            deepCard.bind(
                    deepCardTitle(activeDeepSelection), HrStatusFormatter.deep(status));
            updateWaveformCards(status);
        } catch (Throwable ignored) {
            // Preserve the most recently polled result when the final status read fails.
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
    protected void onStart() {
        super.onStart();
        lifecycleStopped = false;
    }

    @Override
    protected void onStop() {
        lifecycleStopped = true;
        stopCamera();
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        lifecycleStopped = true;
        cameraStartGeneration.destroy();
        cameraStartExecutor.shutdown();
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
        private final int sensorOrientation;

        private CameraEntry(String id, String facing, int sensorOrientation) {
            this.id = id;
            this.facing = facing;
            this.sensorOrientation = sensorOrientation;
        }
    }

    private static final class CameraStartRequest {
        private final String cameraId;
        private final String method;
        private final DeepModelSelection deepSelection;
        private final File modelFile;
        private final File filesDirectory;
        private final AssetManager assets;
        private final File sessionDirectory;
        private final int displayRotationDegrees;

        private CameraStartRequest(
                String cameraId,
                String method,
                DeepModelSelection deepSelection,
                File modelFile,
                File filesDirectory,
                AssetManager assets,
                File sessionDirectory,
                int displayRotationDegrees) {
            this.cameraId = cameraId;
            this.method = method;
            this.deepSelection = deepSelection;
            this.modelFile = modelFile;
            this.filesDirectory = filesDirectory;
            this.assets = assets;
            this.sessionDirectory = sessionDirectory;
            this.displayRotationDegrees = displayRotationDegrees;
        }
    }

    private static String deepCardTitle(DeepModelSelection selection) {
        return selection.enabled() ? "深度 " + selection.displayLabel() : "深度模型";
    }

    private String deepWaveformTitle(DeepModelSelection selection) {
        return selection.enabled()
                ? getString(R.string.waveform_deep_title, selection.displayLabel())
                : getString(R.string.waveform_deep_disabled_title);
    }

    private static String formatDouble(double value) {
        return String.format(Locale.US, "%.4g", value);
    }

    private static File ensureCascadeAsset(File filesDirectory, AssetManager assets)
            throws IOException {
        File cascade = new File(filesDirectory, "haarcascade_frontalface_default.xml");
        if (cascade.isFile() && cascade.length() > 0) {
            return cascade;
        }
        File temporary = new File(cascade.getAbsolutePath() + ".tmp");
        try (InputStream input = assets.open("haarcascade_frontalface_default.xml");
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

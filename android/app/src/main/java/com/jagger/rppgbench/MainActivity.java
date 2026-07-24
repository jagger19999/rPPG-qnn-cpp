package com.jagger.rppgbench;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public final class MainActivity extends Activity {
    private static final int CAMERA_PERMISSION_REQUEST = 1001;
    private static final int ACTION_NONE = 0;
    private static final int ACTION_LIST = 1;
    private static final int ACTION_START = 2;

    private final Handler statusHandler = new Handler(Looper.getMainLooper());
    private final Runnable statusPoll = new Runnable() {
        @Override
        public void run() {
            if (started && nativeHandle != 0) {
                status.setText(NativeBridge.nativeGetStatus(nativeHandle));
                statusHandler.postDelayed(this, 1000);
            }
        }
    };

    private TextView status;
    private Spinner methodSelector;
    private CheckBox deepSelector;
    private String cameraId;
    private long nativeHandle;
    private boolean started;
    private int pendingAction = ACTION_NONE;

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
        ArrayAdapter<String> methodAdapter = new ArrayAdapter<>(
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

        setContentView(content);
        refreshNativeStatus();
    }

    private void refreshNativeStatus() {
        try {
            status.setText(NativeBridge.nativeBuildIdentity());
        } catch (Throwable error) {
            status.setText("NATIVE_LOAD_FAILED: " + error.getClass().getSimpleName());
        }
    }

    private void runWithCameraPermission(int action) {
        if (checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            pendingAction = action;
            requestPermissions(
                    new String[] {Manifest.permission.CAMERA},
                    CAMERA_PERMISSION_REQUEST);
            return;
        }
        performAction(action);
    }

    private void performAction(int action) {
        if (action == ACTION_LIST) {
            listCameras();
        } else if (action == ACTION_START) {
            startCamera();
        }
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
            File outputDirectory = new File(
                    new File(getFilesDir(), "sessions"),
                    "session-" + System.currentTimeMillis());
            File model = new File(
                    new File(getFilesDir(), "models"),
                    "efficientphys_pure.onnx");
            if (deepSelector.isChecked() && !model.isFile()) {
                status.setText(
                        "MODEL_LOAD_FAILED: import models/efficientphys_pure.onnx "
                                + "to app-private storage with adb run-as");
                return;
            }
            String configured = NativeBridge.nativeConfigureProcessing(
                    nativeHandle,
                    methodSelector.getSelectedItem().toString(),
                    cascade.getAbsolutePath(),
                    outputDirectory.getAbsolutePath(),
                    deepSelector.isChecked(),
                    model.getAbsolutePath());
            if (!configured.startsWith("{")) {
                status.setText(configured);
                return;
            }
            String result = NativeBridge.nativeStart(nativeHandle);
            status.setText(result);
            if (!result.contains("\"state\":\"running\"")) {
                return;
            }
            started = true;
            statusHandler.removeCallbacks(statusPoll);
            statusHandler.postDelayed(statusPoll, 1000);
        } catch (Throwable error) {
            status.setText("CAMERA_START_FAILED: " + error.getMessage());
        }
    }

    private void stopCamera() {
        statusHandler.removeCallbacks(statusPoll);
        if (nativeHandle == 0) {
            started = false;
            return;
        }
        try {
            status.setText(NativeBridge.nativeStop(nativeHandle));
        } catch (Throwable error) {
            status.setText("CAMERA_STOP_FAILED: " + error.getMessage());
        } finally {
            started = false;
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
        if (requestCode != CAMERA_PERMISSION_REQUEST) {
            return;
        }
        int action = pendingAction;
        pendingAction = ACTION_NONE;
        if (grantResults.length == 0
                || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            status.setText("CAMERA_PERMISSION_DENIED: permission was not granted");
            return;
        }
        performAction(action);
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

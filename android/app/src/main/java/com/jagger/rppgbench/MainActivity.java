package com.jagger.rppgbench;

import android.app.Activity;
import android.os.Bundle;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

public final class MainActivity extends Activity {
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

        TextView status = new TextView(this);
        content.addView(status);

        Button refresh = new Button(this);
        refresh.setText(R.string.refresh_native_status);
        refresh.setOnClickListener(view -> refreshNativeStatus(status));
        content.addView(refresh);

        setContentView(content);
        refreshNativeStatus(status);
    }

    private static void refreshNativeStatus(TextView status) {
        try {
            status.setText(NativeBridge.nativeBuildIdentity());
        } catch (Throwable error) {
            status.setText("NATIVE_LOAD_FAILED: " + error.getClass().getSimpleName());
        }
    }
}

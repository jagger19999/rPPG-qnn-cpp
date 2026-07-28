package com.jagger.rppgbench.ui;

import android.view.View;
import android.widget.TextView;

import com.jagger.rppgbench.R;

public final class PpgWaveformCard {
    private final TextView title;
    private final TextView status;
    private final PpgWaveformView waveform;

    public PpgWaveformCard(View root) {
        title = root.findViewById(R.id.waveform_title);
        status = root.findViewById(R.id.waveform_status);
        waveform = root.findViewById(R.id.waveform_view);
    }

    public void bind(String titleText, String statusText, PpgWaveformSnapshot snapshot) {
        title.setText(titleText);
        status.setText(statusText);
        waveform.setSnapshot(snapshot);
    }

    public void clear(String titleText, String statusText) {
        bind(titleText, statusText, null);
    }
}

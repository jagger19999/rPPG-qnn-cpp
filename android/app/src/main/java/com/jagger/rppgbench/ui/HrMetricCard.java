package com.jagger.rppgbench.ui;

import android.graphics.Color;
import android.view.View;
import android.widget.TextView;

import com.jagger.rppgbench.R;

public final class HrMetricCard {
    private final View accent;
    private final TextView title;
    private final TextView primary;
    private final TextView secondary;

    public HrMetricCard(View root, int accentColor) {
        accent = root.findViewById(R.id.card_accent);
        title = root.findViewById(R.id.card_title);
        primary = root.findViewById(R.id.card_primary);
        secondary = root.findViewById(R.id.card_secondary);
        if (accent != null) {
            accent.setBackgroundColor(accentColor);
        }
    }

    public void bind(String titleText, HrStatusFormatter.CardView data) {
        title.setText(titleText);
        primary.setText(data.primary);
        secondary.setText(data.secondary);
    }
}

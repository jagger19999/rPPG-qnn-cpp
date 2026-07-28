package com.jagger.rppgbench.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.AttributeSet;
import android.view.View;

import java.util.Locale;

public final class PpgWaveformView extends View {
    private static final int VALID_COLOR = Color.rgb(47, 143, 116);
    private static final int INVALID_COLOR = Color.rgb(184, 107, 43);
    private final Paint gridPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Path path = new Path();
    private PpgWaveformSnapshot snapshot;
    private float[] values = new float[0];
    private float[] points = new float[0];

    public PpgWaveformView(Context context, AttributeSet attrs) {
        super(context, attrs);
        gridPaint.setColor(Color.rgb(220, 220, 214));
        gridPaint.setStrokeWidth(dp(1));
        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeWidth(dp(2));
        textPaint.setColor(Color.rgb(102, 102, 102));
        textPaint.setTextSize(dp(11));
    }

    public void setSnapshot(PpgWaveformSnapshot value) {
        if (snapshot != null && value != null && snapshot.revision == value.revision) {
            return;
        }
        snapshot = value;
        values = value == null ? new float[0] : value.values();
        points = new float[values.length * 2];
        invalidate();
    }

    public void clear() {
        setSnapshot(null);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float left = dp(8);
        float right = getWidth() - dp(8);
        float top = dp(8);
        float bottom = getHeight() - dp(22);
        float middle = (top + bottom) / 2f;
        canvas.drawLine(left, middle, right, middle, gridPaint);
        canvas.drawLine(left, top, right, top, gridPaint);
        canvas.drawLine(left, bottom, right, bottom, gridPaint);
        if (snapshot == null) {
            return;
        }
        PpgWaveformGeometry.fillPoints(values, left, top, right - left, bottom - top, points);
        path.reset();
        path.moveTo(points[0], points[1]);
        for (int index = 2; index < points.length; index += 2) {
            path.lineTo(points[index], points[index + 1]);
        }
        linePaint.setColor(snapshot.valid ? VALID_COLOR : INVALID_COLOR);
        canvas.drawPath(path, linePaint);
        canvas.drawText(String.format(Locale.US, "%.1f s", snapshot.relativeStartSeconds()),
                left, getHeight() - dp(5), textPaint);
        String end = "0 s";
        canvas.drawText(end, right - textPaint.measureText(end), getHeight() - dp(5), textPaint);
    }

    private float dp(float value) {
        return value * getResources().getDisplayMetrics().density;
    }
}

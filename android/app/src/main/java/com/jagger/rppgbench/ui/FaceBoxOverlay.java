package com.jagger.rppgbench.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

public final class FaceBoxOverlay extends View {
    private static final float STROKE_WIDTH_DP = 3f;

    private final Paint strokePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private boolean hasFaceRect;
    private boolean mirrorHorizontal;
    private float normalizedX;
    private float normalizedY;
    private float normalizedWidth;
    private float normalizedHeight;

    public FaceBoxOverlay(Context context) {
        super(context);
        init();
    }

    public FaceBoxOverlay(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public FaceBoxOverlay(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        strokePaint.setStyle(Paint.Style.STROKE);
        strokePaint.setStrokeWidth(STROKE_WIDTH_DP * getResources().getDisplayMetrics().density);
        strokePaint.setColor(0xFF4CAF50);
    }

    public void setMirrorHorizontal(boolean mirrorHorizontal) {
        if (this.mirrorHorizontal == mirrorHorizontal) {
            return;
        }
        this.mirrorHorizontal = mirrorHorizontal;
        invalidate();
    }

    public void setFaceRect(float x, float y, float width, float height) {
        normalizedX = x;
        normalizedY = y;
        normalizedWidth = width;
        normalizedHeight = height;
        hasFaceRect = true;
        invalidate();
    }

    public void clearFaceRect() {
        hasFaceRect = false;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (!hasFaceRect || getWidth() <= 0 || getHeight() <= 0) {
            return;
        }
        float left = normalizedX * getWidth();
        float top = normalizedY * getHeight();
        float right = left + normalizedWidth * getWidth();
        float bottom = top + normalizedHeight * getHeight();
        if (mirrorHorizontal) {
            float mirroredLeft = getWidth() - right;
            float mirroredRight = getWidth() - left;
            left = mirroredLeft;
            right = mirroredRight;
        }
        canvas.drawRect(left, top, right, bottom, strokePaint);
    }
}

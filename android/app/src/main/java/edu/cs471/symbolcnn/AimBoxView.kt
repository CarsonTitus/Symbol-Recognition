package edu.cs471.symbolcnn

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/**
 * AimBoxView - draws the square aiming box in the middle of the screen.
 *
 * Everything outside the box is dimmed. Only the picture INSIDE the box is sent to the
 * model, so the user's aiming does the job of the future "Step 1" detector: it hands the
 * model a close-up crop of one shape, just like the training images (CLAUDE.md §1).
 */
class AimBoxView(context: Context, attrs: AttributeSet? = null) : View(context, attrs) {

    companion object {
        /** Box side as a share of the screen's shorter side. MainActivity crops the same square. */
        const val BOX_FRACTION = 0.7f

        /** The box, in screen pixels, for a view of the given size (centered). */
        fun boxFor(viewWidth: Int, viewHeight: Int): RectF {
            val side = BOX_FRACTION * min(viewWidth, viewHeight)
            val left = (viewWidth - side) / 2f
            val top = (viewHeight - side) / 2f
            return RectF(left, top, left + side, top + side)
        }
    }

    private val dim = Paint().apply { color = Color.argb(110, 0, 0, 0) }
    private val outline = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val box = boxFor(width, height)
        // Four dark rectangles around the box (top, bottom, left, right).
        canvas.drawRect(0f, 0f, width.toFloat(), box.top, dim)
        canvas.drawRect(0f, box.bottom, width.toFloat(), height.toFloat(), dim)
        canvas.drawRect(0f, box.top, box.left, box.bottom, dim)
        canvas.drawRect(box.right, box.top, width.toFloat(), box.bottom, dim)
        canvas.drawRoundRect(box, 24f, 24f, outline)
    }
}

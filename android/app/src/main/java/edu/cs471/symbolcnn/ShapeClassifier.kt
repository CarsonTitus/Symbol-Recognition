package edu.cs471.symbolcnn

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * ShapeClassifier - runs our trained CNN (symbolcnn.tflite) on one square picture.
 *
 * The model file, its class names and settings come from app/src/main/assets/, put there by
 *     .venv-export/bin/python scripts/export_tflite.py --run latest --android
 * The model already contains the color scaling and the softmax (see export_tflite.py), so
 * this class only has to: shrink the picture to 224x224, turn pixels into numbers 0-1,
 * run the model, and apply the confidence threshold (CLAUDE.md §3).
 */
class ShapeClassifier(context: Context) : Closeable {

    /** One answer from the model. */
    data class Result(
        val probabilities: FloatArray,   // one per class, adds up to 1
        val answer: String,              // the class name after the confidence threshold
        val confidence: Float,           // the top probability
        val unsure: Boolean,             // true if the threshold turned the answer into unknown_symbol
        val millis: Long,                // how long the model took
    )

    val labels: List<String> = context.assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }
    private val info = JSONObject(context.assets.open("model_info.json").bufferedReader().readText())
    val threshold: Float = info.getDouble("confidence_threshold").toFloat()
    val modelRun: String = info.getString("run")
    private val size: Int = info.getInt("input_size")

    private val interpreter: Interpreter
    // Reused every frame so the phone doesn't create garbage 10+ times a second.
    private val input = ByteBuffer.allocateDirect(4 * size * size * 3).order(ByteOrder.nativeOrder())
    private val pixels = IntArray(size * size)
    private val output = Array(1) { FloatArray(labels.size) }

    init {
        // Load the model file into memory and start the LiteRT interpreter (4 CPU threads).
        val bytes = context.assets.open("symbolcnn.tflite").use { it.readBytes() }
        val model = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        model.put(bytes).rewind()
        interpreter = Interpreter(model, Interpreter.Options().setNumThreads(4))
    }

    fun classify(square: Bitmap): Result {
        // 1. Shrink to 224x224 ("filter = true" smooths, so thin lines don't break up).
        val small = Bitmap.createScaledBitmap(square, size, size, true)
        small.getPixels(pixels, 0, size, 0, 0, size, size)

        // 2. Pixels -> numbers 0-1, in the order the model expects: row by row, red-green-blue.
        input.rewind()
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF) / 255f)   // red
            input.putFloat(((p shr 8) and 0xFF) / 255f)    // green
            input.putFloat((p and 0xFF) / 255f)            // blue
        }
        input.rewind()

        // 3. Run the model.
        val start = SystemClock.elapsedRealtime()
        interpreter.run(input, output)
        val millis = SystemClock.elapsedRealtime() - start
        val probs = output[0].copyOf()

        // 4. Confidence threshold: not sure enough -> unknown_symbol (the safety net).
        val best = probs.indices.maxBy { probs[it] }
        val unsure = probs[best] < threshold
        val answer = if (unsure) "unknown_symbol" else labels[best]
        return Result(probs, answer, probs[best], unsure, millis)
    }

    override fun close() = interpreter.close()
}

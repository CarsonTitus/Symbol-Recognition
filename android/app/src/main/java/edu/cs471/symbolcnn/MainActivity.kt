package edu.cs471.symbolcnn

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.util.Size
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * MainActivity - the app's one screen: live camera, aiming box, and the model's answer.
 *
 * Every camera frame (as fast as the model keeps up):
 *   1. turn the frame upright (phones deliver frames sideways),
 *   2. cut out the part that is inside the aiming box on screen,
 *   3. ask ShapeClassifier what shape it is,
 *   4. smooth the answer over the last 5 frames and show it.
 * Everything runs on the phone; no internet is needed.
 */
class MainActivity : ComponentActivity() {

    private lateinit var preview: PreviewView
    private lateinit var answerText: TextView
    private lateinit var detailsText: TextView
    private var classifier: ShapeClassifier? = null
    private val smoother = AnswerSmoother()
    private val worker: ExecutorService = Executors.newSingleThreadExecutor()   // model runs off the screen thread

    private val askForCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else answerText.text = "Camera permission needed"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        preview = findViewById(R.id.preview)
        answerText = findViewById(R.id.answer)
        detailsText = findViewById(R.id.details)

        classifier = try {
            ShapeClassifier(this)
        } catch (e: Exception) {
            answerText.text = "No model in the app"
            detailsText.text = "Run scripts/export_tflite.py --android, then rebuild.\n${e.message}"
            null
        }
        if (classifier == null) return

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            askForCamera.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()

            // Preview and analysis both use 4:3 pictures, so the box on screen and the crop match.
            val fourByThree = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .build()
            val screenPreview = Preview.Builder().setResolutionSelector(fourByThree).build()
            screenPreview.surfaceProvider = preview.surfaceProvider

            // Analysis frames are bigger than the old app's 640x480 so pen lines stay visible.
            val bigFrames = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(1600, 1200),
                    ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(bigFrames)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)   // skip frames we can't keep up with
                .build()
            analysis.setAnalyzer(worker, ::analyze)

            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, screenPreview, analysis)
        }, ContextCompat.getMainExecutor(this))
    }

    /** Runs on the worker thread for each camera frame. */
    private fun analyze(frame: ImageProxy) {
        val model = classifier ?: return frame.close()
        val upright = frame.use { rotate(it.toBitmap(), it.imageInfo.rotationDegrees) }
        val square = cropAimBox(upright) ?: return
        val result = model.classify(square)
        val shown = smoother.add(result.answer)
        runOnUiThread { show(shown, result, model) }
    }

    /** Phones store frames sideways; turn the picture the way the user sees it. */
    private fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return bitmap
        val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Cut out exactly the part of the camera picture that is inside the aiming box.
     * The preview is stretched to fill the screen and its edges trimmed (FILL_CENTER):
     * one camera pixel covers `scale` screen pixels, so the box is (box width / scale) camera pixels wide.
     */
    private fun cropAimBox(picture: Bitmap): Bitmap? {
        val viewWidth = preview.width
        val viewHeight = preview.height
        if (viewWidth == 0 || viewHeight == 0) return null
        val scale = max(viewWidth.toFloat() / picture.width, viewHeight.toFloat() / picture.height)
        val box = AimBoxView.boxFor(viewWidth, viewHeight)
        val side = min((box.width() / scale).roundToInt(), min(picture.width, picture.height))
        val left = (picture.width - side) / 2      // the box is centered on screen,
        val top = (picture.height - side) / 2      // so it is centered in the picture too
        return Bitmap.createBitmap(picture, left, top, side, side)
    }

    private fun show(shown: String?, result: ShapeClassifier.Result, model: ShapeClassifier) {
        val name = shown ?: result.answer
        val isShape = name != "unknown_symbol"
        val percent = (result.confidence * 100).roundToInt()
        answerText.text = if (isShape) "${name.uppercase()}  $percent%" else "UNKNOWN"
        answerText.setTextColor(if (isShape) Color.rgb(0, 255, 102) else Color.LTGRAY)   // green = a named shape
        val probs = model.labels.indices.joinToString("  ") { i ->
            val short = model.labels[i].removeSuffix("_symbol").take(5)
            "$short ${(result.probabilities[i] * 100).roundToInt()}%"
        }
        val unsure = if (result.unsure) "  (unsure)" else ""
        detailsText.text = "$probs\n${result.millis} ms per frame$unsure"
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.shutdown()
        classifier?.close()
    }
}

// app/build.gradle.kts - how to build the demo app and which libraries it uses.
plugins {
    id("com.android.application")
}

android {
    namespace = "edu.cs471.symbolcnn"
    compileSdk = 37

    defaultConfig {
        applicationId = "edu.cs471.symbolcnn"
        minSdk = 26                 // Android 8.0 or newer
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    // Activity: the app's single screen, plus the camera-permission request.
    implementation("androidx.activity:activity-ktx:1.13.0")
    implementation("androidx.core:core-ktx:1.19.1")

    // CameraX: Google's camera library. "view" gives the live preview on screen,
    // "lifecycle" starts/stops the camera with the app, "camera2" talks to the hardware.
    val cameraX = "1.6.2"
    implementation("androidx.camera:camera-core:$cameraX")
    implementation("androidx.camera:camera-camera2:$cameraX")
    implementation("androidx.camera:camera-lifecycle:$cameraX")
    implementation("androidx.camera:camera-view:$cameraX")

    // LiteRT (formerly TensorFlow Lite): runs our .tflite model on the phone, offline.
    implementation("com.google.ai.edge.litert:litert:1.4.2")
}

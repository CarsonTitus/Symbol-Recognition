// build.gradle.kts (top level) - which version of the Android build plugin to use.
// AGP = Android Gradle Plugin. Version 9 compiles Kotlin by itself (no separate Kotlin plugin).
plugins {
    id("com.android.application") version "9.5.2" apply false
}

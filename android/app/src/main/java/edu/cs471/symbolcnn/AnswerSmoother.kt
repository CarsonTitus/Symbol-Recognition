package edu.cs471.symbolcnn

/**
 * AnswerSmoother - stops the on-screen answer from flickering.
 *
 * The model answers many times a second, and a shaky hand can make single frames wrong.
 * The shown answer only changes when the same answer appears in at least `needed` of the
 * last `window` frames (3 of 5: the same rule planned for the G2 glasses, CLAUDE.md §12).
 */
class AnswerSmoother(private val window: Int = 5, private val needed: Int = 3) {
    private val recent = ArrayDeque<String>()
    var shown: String? = null
        private set

    /** Add the newest frame's answer; returns the answer to show. */
    fun add(answer: String): String? {
        recent.addLast(answer)
        if (recent.size > window) recent.removeFirst()
        val (mostCommon, count) = recent.groupingBy { it }.eachCount().maxBy { it.value }.toPair()
        if (count >= needed) shown = mostCommon
        return shown
    }
}

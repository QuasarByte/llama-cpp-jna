package com.quasarbyte.llama.cpp.jna.binding.logging;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.sun.jna.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Custom logger for llama.cpp log messages.
 * <p>
 * This class provides a bridge between llama.cpp's native logging system and
 * Java's SLF4J logging framework. It captures all log messages from the native
 * library and routes them through the appropriate SLF4J log levels.
 *
 * <p><strong>Usage:</strong></p>
 * <pre>{@code
 * LlamaLibrary llamaLibrary = new LlamaLibraryFactory().getInstance();
 * CustomLlamaLoggerBinding.setupCustomLogging(llamaLibrary);
 * }</pre>
 */
public class CustomLlamaLoggerBinding {

    private static final Logger logger = LoggerFactory.getLogger(CustomLlamaLoggerBinding.class);

    // Keep a strong reference to prevent garbage collection
    private static LlamaLibrary.LlamaLogCallback activeCallback;

    /**
     * Setup custom logging for llama.cpp.
     * <p>
     * This method registers a callback with the llama.cpp library to intercept
     * all log messages and route them through SLF4J with appropriate log levels.
     *
     * @param llamaLibrary the llama library instance
     */
    public static void setupCustomLogging(LlamaLibrary llamaLibrary) {
        if (llamaLibrary == null) {
            logger.warn("Cannot setup custom logging: llamaLibrary is null");
            return;
        }

        logger.info("Setting up custom llama.cpp logging");

        // Create the callback implementation
        activeCallback = new LlamaLibrary.LlamaLogCallback() {
            @Override
            public void invoke(int level, String text, Pointer userData) {
                handleLogMessage(level, text, userData);
            }
        };

        // Register the callback with llama.cpp
        llamaLibrary.llama_log_set(activeCallback, null);

        logger.info("Custom llama.cpp logging enabled");
    }

    /**
     * Disable custom logging and restore default llama.cpp stderr output.
     *
     * @param llamaLibrary the llama library instance
     */
    public static void disableCustomLogging(LlamaLibrary llamaLibrary) {
        if (llamaLibrary == null) {
            logger.warn("Cannot disable custom logging: llamaLibrary is null");
            return;
        }

        logger.info("Disabling custom llama.cpp logging");

        // Clear the callback to restore default stderr output
        llamaLibrary.llama_log_set(null, null);
        activeCallback = null;

        logger.info("Custom llama.cpp logging disabled");
    }

    /**
     * Handle a log message from llama.cpp.
     * <p>
     * This method processes the native log message and routes it to the
     * appropriate SLF4J log level based on the GGML log level.
     *
     * @param level the GGML log level constant
     * @param text the log message text
     * @param userData user data pointer (unused)
     */
    private static void handleLogMessage(int level, String text, Pointer userData) {
        // Remove trailing newlines for cleaner logging
        String cleanText = text != null ? text.trim() : "";

        // Skip empty messages
        if (cleanText.isEmpty()) {
            return;
        }

        // Route to appropriate log level
        switch (level) {
            case LlamaLibrary.GGML_LOG_LEVEL_DEBUG:
                logger.debug("[llama.cpp] {}", cleanText);
                break;

            case LlamaLibrary.GGML_LOG_LEVEL_INFO:
                logger.info("[llama.cpp] {}", cleanText);
                break;

            case LlamaLibrary.GGML_LOG_LEVEL_WARN:
                logger.warn("[llama.cpp] {}", cleanText);
                break;

            case LlamaLibrary.GGML_LOG_LEVEL_ERROR:
                logger.error("[llama.cpp] {}", cleanText);
                break;

            case LlamaLibrary.GGML_LOG_LEVEL_CONT:
                // Continue previous log - treat as debug level
                logger.debug("[llama.cpp] {}", cleanText);
                break;

            case LlamaLibrary.GGML_LOG_LEVEL_NONE:
                // Should not happen, but log as debug if it does
                logger.debug("[llama.cpp] (none) {}", cleanText);
                break;

            default:
                // Unknown log level - log as info with level indicator
                logger.info("[llama.cpp] (level={}) {}", level, cleanText);
                break;
        }
    }

    /**
     * Get a description of the GGML log level.
     *
     * @param level the GGML log level constant
     * @return human-readable description of the log level
     */
    public static String getLogLevelDescription(int level) {
        switch (level) {
            case LlamaLibrary.GGML_LOG_LEVEL_NONE:
                return "NONE";
            case LlamaLibrary.GGML_LOG_LEVEL_DEBUG:
                return "DEBUG";
            case LlamaLibrary.GGML_LOG_LEVEL_INFO:
                return "INFO";
            case LlamaLibrary.GGML_LOG_LEVEL_WARN:
                return "WARN";
            case LlamaLibrary.GGML_LOG_LEVEL_ERROR:
                return "ERROR";
            case LlamaLibrary.GGML_LOG_LEVEL_CONT:
                return "CONT";
            default:
                return "UNKNOWN(" + level + ")";
        }
    }

    /**
     * Check if custom logging is currently active.
     *
     * @return true if custom logging callback is registered
     */
    public static boolean isCustomLoggingActive() {
        return activeCallback != null;
    }
}
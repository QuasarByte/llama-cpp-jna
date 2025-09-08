package com.quasarbyte.llama.cpp.jna.exception;

public class LlamaCppJnaException extends RuntimeException {
    public LlamaCppJnaException() {
    }

    public LlamaCppJnaException(String message) {
        super(message);
    }

    public LlamaCppJnaException(String message, Throwable cause) {
        super(message, cause);
    }

    public LlamaCppJnaException(Throwable cause) {
        super(cause);
    }
}

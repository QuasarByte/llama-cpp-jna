package com.quasarbyte.llama.cpp.jna.exception;

public class LlamaFunctionCallResultException extends LlamaCppJnaException {

    public LlamaFunctionCallResultException() {
    }

    public LlamaFunctionCallResultException(String message) {
        super(message);
    }

    public LlamaFunctionCallResultException(String message, Throwable cause) {
        super(message, cause);
    }

    public LlamaFunctionCallResultException(Throwable cause) {
        super(cause);
    }
}

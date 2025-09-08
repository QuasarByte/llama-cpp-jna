package com.quasarbyte.llama.cpp.jna.exception;

public class LlamaFunctionCallLongResultException extends LlamaCppJnaException {

    private final long result;

    public LlamaFunctionCallLongResultException(long result) {
        super(String.format("Llama function call long result exception, result: %d", result));
        this.result = result;
    }

    public LlamaFunctionCallLongResultException(long result, String message) {
        super(String.format(message, result));
        this.result = result;
    }

    public LlamaFunctionCallLongResultException(long result, String message, Throwable cause) {
        super(message, cause);
        this.result = result;
    }

    public LlamaFunctionCallLongResultException(long result, Throwable cause) {
        super(cause);
        this.result = result;
    }

    public long getResult() {
        return result;
    }
}

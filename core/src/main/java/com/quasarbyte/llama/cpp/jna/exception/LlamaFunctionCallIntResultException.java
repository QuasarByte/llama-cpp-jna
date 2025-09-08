package com.quasarbyte.llama.cpp.jna.exception;

public class LlamaFunctionCallIntResultException extends LlamaCppJnaException {

    private final int result;

    public LlamaFunctionCallIntResultException(int result) {
        super(String.format("Llama function call int result exception, result: %d", result));
        this.result = result;
    }

    public LlamaFunctionCallIntResultException(int result, String message) {
        super(String.format(message, result));
        this.result = result;
    }

    public LlamaFunctionCallIntResultException(int result, String message, Throwable cause) {
        super(message, cause);
        this.result = result;
    }

    public LlamaFunctionCallIntResultException(int result, Throwable cause) {
        super(cause);
        this.result = result;
    }

    public int getResult() {
        return result;
    }
}

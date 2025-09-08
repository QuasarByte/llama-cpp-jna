package com.quasarbyte.llama.cpp.jna.model.library;

public class LlamaContext {
    private final LlamaContextNative contextPointer;

    public LlamaContext(LlamaContextNative contextPointer) {
        this.contextPointer = contextPointer;
    }

    public LlamaContextNative getContextPointer() {
        return contextPointer;
    }

}
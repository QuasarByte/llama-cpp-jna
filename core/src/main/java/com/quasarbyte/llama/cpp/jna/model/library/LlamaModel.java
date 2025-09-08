package com.quasarbyte.llama.cpp.jna.model.library;

public class LlamaModel {
    private LlamaModelNative modelPointer;

    public LlamaModelNative getModelPointer() {
        return modelPointer;
    }

    public LlamaModel setModelPointer(LlamaModelNative modelPointer) {
        this.modelPointer = modelPointer;
        return this;
    }
}
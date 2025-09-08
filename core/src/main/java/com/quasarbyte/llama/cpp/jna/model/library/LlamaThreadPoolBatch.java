package com.quasarbyte.llama.cpp.jna.model.library;

public class LlamaThreadPoolBatch {
    private final LlamaThreadPoolBatchNative llamaThreadPoolNative;

    public LlamaThreadPoolBatch(LlamaThreadPoolBatchNative llamaThreadPoolNative) {
        this.llamaThreadPoolNative = llamaThreadPoolNative;
    }

    public LlamaThreadPoolBatchNative getLlamaThreadPoolNative() {
        return llamaThreadPoolNative;
    }
}

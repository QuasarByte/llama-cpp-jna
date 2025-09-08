package com.quasarbyte.llama.cpp.jna.model.library;

public class LlamaThreadPool {
    private final LlamaThreadPoolNative llamaThreadPoolNative;

    public LlamaThreadPool(LlamaThreadPoolNative llamaThreadPoolNative) {
        this.llamaThreadPoolNative = llamaThreadPoolNative;
    }

    public LlamaThreadPoolNative getLlamaThreadPoolNative() {
        return llamaThreadPoolNative;
    }
}

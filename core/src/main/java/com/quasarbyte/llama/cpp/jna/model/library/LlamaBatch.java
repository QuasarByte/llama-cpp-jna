package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Memory;

// Wrapper class for LlamaBatchNative to manage memory
public class LlamaBatch {
    private LlamaBatchNative nativeBatch;
    private Memory tokensMemory;

    public LlamaBatchNative getNativeBatch() {
        return nativeBatch;
    }

    public LlamaBatch setNativeBatch(LlamaBatchNative nativeBatch) {
        this.nativeBatch = nativeBatch;
        return this;
    }

    public Memory getTokensMemory() {
        return tokensMemory;
    }

    public LlamaBatch setTokensMemory(Memory tokensMemory) {
        this.tokensMemory = tokensMemory;
        return this;
    }

    public int getTokenCount() {
        return nativeBatch != null ? nativeBatch.n_tokens : 0;
    }
}

package com.quasarbyte.llama.cpp.jna.model.library;

import com.quasarbyte.llama.cpp.jna.model.library.ggml.GgmlBackendBufferTypeNative;

/**
 * Plain Java helper mirroring {@code llama_model_tensor_buft_override}.
 */
public class LlamaModelTensorBuftOverride {

    private String pattern;
    private GgmlBackendBufferTypeNative bufferType;

    public LlamaModelTensorBuftOverride() {
    }

    public LlamaModelTensorBuftOverride(String pattern, GgmlBackendBufferTypeNative bufferType) {
        this.pattern = pattern;
        this.bufferType = bufferType;
    }

    public String getPattern() {
        return pattern;
    }

    public LlamaModelTensorBuftOverride setPattern(String pattern) {
        this.pattern = pattern;
        return this;
    }

    public GgmlBackendBufferTypeNative getBufferType() {
        return bufferType;
    }

    public LlamaModelTensorBuftOverride setBufferType(GgmlBackendBufferTypeNative bufferType) {
        this.bufferType = bufferType;
        return this;
    }

    LlamaModelTensorBuftOverrideNative toNative() {
        LlamaModelTensorBuftOverrideNative nativeOverride = new LlamaModelTensorBuftOverrideNative();
        nativeOverride.setPattern(pattern);
        nativeOverride.bufferType = bufferType;
        nativeOverride.write();
        return nativeOverride;
    }
}

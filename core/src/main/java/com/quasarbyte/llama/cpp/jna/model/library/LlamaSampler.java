package com.quasarbyte.llama.cpp.jna.model.library;

public class LlamaSampler {
    private LlamaSamplerNative samplerPointer;
    private ChainRole chainRole;

    public LlamaSamplerNative getSamplerPointer() {
        return samplerPointer;
    }

    public LlamaSampler setSamplerPointer(LlamaSamplerNative samplerPointer) {
        this.samplerPointer = samplerPointer;
        return this;
    }

    public ChainRole getChainRole() {
        return chainRole;
    }

    public LlamaSampler setChainRole(ChainRole chainRole) {
        this.chainRole = chainRole;
        return this;
    }
}
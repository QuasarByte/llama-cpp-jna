package com.quasarbyte.llama.cpp.jna.model.library;

/**
 * Plain Java representation of {@code llama_token_data} for easier debugging and tests.
 */
public class LlamaTokenData {
    private int id;
    private float logit;
    private float probability;

    public LlamaTokenData() {
    }

    public LlamaTokenData(int id, float logit, float probability) {
        this.id = id;
        this.logit = logit;
        this.probability = probability;
    }

    public int getId() {
        return id;
    }

    public LlamaTokenData setId(int id) {
        this.id = id;
        return this;
    }

    public float getLogit() {
        return logit;
    }

    public LlamaTokenData setLogit(float logit) {
        this.logit = logit;
        return this;
    }

    public float getProbability() {
        return probability;
    }

    public LlamaTokenData setProbability(float probability) {
        this.probability = probability;
        return this;
    }

    LlamaTokenDataNative toNative() {
        return new LlamaTokenDataNative(id, logit, probability);
    }

    static LlamaTokenData fromNative(LlamaTokenDataNative nativeEntry) {
        return new LlamaTokenData(nativeEntry.id, nativeEntry.logit, nativeEntry.p);
    }
}

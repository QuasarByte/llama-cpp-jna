package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import java.util.Arrays;
import java.util.List;

/**
 * JNA mapping for the native llama_logit_bias structure.
 * <p>
 * typedef struct llama_logit_bias {
 *     llama_token token;
 *     float bias;
 * } llama_logit_bias;
 */
@Structure.FieldOrder({"token", "bias"})
public class LlamaLogitBiasNative extends Structure {

    public int token;
    public float bias;

    public LlamaLogitBiasNative() {
        super();
    }

    public LlamaLogitBiasNative(int token, float bias) {
        super();
        this.token = token;
        this.bias = bias;
    }

    public LlamaLogitBiasNative(Pointer pointer) {
        super(pointer);
    }

    public int getToken() {
        return token;
    }

    public LlamaLogitBiasNative setToken(int token) {
        this.token = token;
        return this;
    }

    public float getBias() {
        return bias;
    }

    public LlamaLogitBiasNative setBias(float bias) {
        this.bias = bias;
        return this;
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("token", "bias");
    }

    public static class ByReference extends LlamaLogitBiasNative implements Structure.ByReference {
        public ByReference() {}
        public ByReference(Pointer pointer) { super(pointer); }
    }

    public static class ByValue extends LlamaLogitBiasNative implements Structure.ByValue {
        public ByValue() {}
    }
}

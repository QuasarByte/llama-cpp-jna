package com.quasarbyte.llama.cpp.jna.model.library;

import com.quasarbyte.llama.cpp.jna.model.library.ggml.GgmlBackendBufferTypeNative;
import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

/**
 * JNA mapping of {@code llama_model_tensor_buft_override}.
 */
@Structure.FieldOrder({"pattern", "bufferType"})
public class LlamaModelTensorBuftOverrideNative extends Structure {

    /** glob-like pattern for tensor names */
    public Pointer pattern;

    /** buffer type override */
    public GgmlBackendBufferTypeNative bufferType;

    private transient String cachedPattern;
    private transient Memory ownedPatternMemory;

    public LlamaModelTensorBuftOverrideNative() {
        super();
    }

    public LlamaModelTensorBuftOverrideNative(Pointer pointer) {
        super(pointer);
        read();
    }

    public String getPattern() {
        if (pattern == null || Pointer.NULL.equals(pattern)) {
            return null;
        }
        if (cachedPattern == null) {
            cachedPattern = pattern.getString(0);
        }
        return cachedPattern;
    }

    public void setPattern(String value) {
        cachedPattern = value;
        if (ownedPatternMemory != null) {
            ownedPatternMemory.close();
            ownedPatternMemory = null;
        }
        if (value == null) {
            pattern = Pointer.NULL;
            return;
        }
        ownedPatternMemory = new Memory(value.length() + 1L);
        ownedPatternMemory.setString(0, value);
        pattern = ownedPatternMemory;
    }

    public static class ByReference extends LlamaModelTensorBuftOverrideNative implements Structure.ByReference {
        public ByReference() {
            super();
        }

        public ByReference(Pointer pointer) {
            super(pointer);
        }
    }
}

package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

/**
 * JNA mapping of {@code llama_token_data}.
 */
@Structure.FieldOrder({"id", "logit", "p"})
public class LlamaTokenDataNative extends Structure {

    /** token id (llama_token) */
    public int id;

    /** token logit value */
    public float logit;

    /** token probability */
    public float p;

    public LlamaTokenDataNative() {
        super();
    }

    public LlamaTokenDataNative(Pointer pointer) {
        super(pointer);
        read();
    }

    /** convenience constructor */
    public LlamaTokenDataNative(int id, float logit, float probability) {
        this();
        this.id = id;
        this.logit = logit;
        this.p = probability;
    }

    public static class ByReference extends LlamaTokenDataNative implements Structure.ByReference {
        public ByReference() {
            super();
        }

        public ByReference(Pointer pointer) {
            super(pointer);
        }
    }

    public static class ByValue extends LlamaTokenDataNative implements Structure.ByValue {
        public ByValue() {
            super();
        }
    }
}

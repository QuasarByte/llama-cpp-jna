package com.quasarbyte.llama.cpp.jna.model.library;

import com.quasarbyte.llama.cpp.jna.model.library.llama.LlamaModelKvOverrideType;
import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import com.sun.jna.Union;

/**
 * Native mapping for {@code llama_model_kv_override}.
 */
@Structure.FieldOrder({"tag", "key", "value"})
public class LlamaModelKvOverrideNative extends Structure {

    public int tag;
    public byte[] key = new byte[128];
    public ValueUnion value;

    private transient Memory stringMemory;

    public LlamaModelKvOverrideNative() {
        super();
    }

    public LlamaModelKvOverrideNative(Pointer pointer) {
        super(pointer);
        read();
    }

    public void setKey(String keyValue) {
        byte[] bytes = keyValue == null ? new byte[0] : keyValue.getBytes();
        int length = Math.min(bytes.length, key.length - 1);
        System.arraycopy(bytes, 0, key, 0, length);
        key[length] = 0; // null terminator
    }

    public String getKey() {
        int len = 0;
        while (len < key.length && key[len] != 0) {
            len++;
        }
        return new String(key, 0, len);
    }

    public void setStringValue(String valueStr) {
        if (stringMemory != null) {
            stringMemory.close();
            stringMemory = null;
        }
        stringMemory = new Memory(valueStr.length() + 1L);
        stringMemory.setString(0, valueStr);
        value.setType(byte[].class);
        byte[] str = new byte[128];
        byte[] bytes = valueStr.getBytes();
        int length = Math.min(bytes.length, str.length - 1);
        System.arraycopy(bytes, 0, str, 0, length);
        str[length] = 0;
        value.val_str = str;
    }

    public static class ValueUnion extends Union {
        public long val_i64;
        public double val_f64;
        public byte val_bool;
        public byte[] val_str = new byte[128];

        public ValueUnion() {
            super();
        }

        public ValueUnion(Pointer pointer) {
            super(pointer);
            read();
        }
    }

    public static class ByReference extends LlamaModelKvOverrideNative implements Structure.ByReference {
        public ByReference() {
            super();
        }

        public ByReference(Pointer pointer) {
            super(pointer);
        }
    }

    public LlamaModelKvOverrideType getType() {
        return LlamaModelKvOverrideType.fromValue(tag);
    }
}

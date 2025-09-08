package com.quasarbyte.llama.cpp.jna.model.library;

import com.quasarbyte.llama.cpp.jna.model.library.llama.LlamaModelKvOverrideType;

/**
 * Plain Java representation of {@code llama_model_kv_override}.
 */
public class LlamaModelKvOverride {

    private String key;
    private LlamaModelKvOverrideType type = LlamaModelKvOverrideType.LLAMA_KV_OVERRIDE_TYPE_INT;
    private long longValue;
    private double doubleValue;
    private boolean booleanValue;
    private String stringValue;

    public String getKey() {
        return key;
    }

    public LlamaModelKvOverride setKey(String key) {
        this.key = key;
        return this;
    }

    public LlamaModelKvOverrideType getType() {
        return type;
    }

    public LlamaModelKvOverride setType(LlamaModelKvOverrideType type) {
        this.type = type;
        return this;
    }

    public long getLongValue() {
        return longValue;
    }

    public LlamaModelKvOverride setLongValue(long longValue) {
        this.longValue = longValue;
        return this;
    }

    public double getDoubleValue() {
        return doubleValue;
    }

    public LlamaModelKvOverride setDoubleValue(double doubleValue) {
        this.doubleValue = doubleValue;
        return this;
    }

    public boolean isBooleanValue() {
        return booleanValue;
    }

    public LlamaModelKvOverride setBooleanValue(boolean booleanValue) {
        this.booleanValue = booleanValue;
        return this;
    }

    public String getStringValue() {
        return stringValue;
    }

    public LlamaModelKvOverride setStringValue(String stringValue) {
        this.stringValue = stringValue;
        return this;
    }

    LlamaModelKvOverrideNative toNative() {
        LlamaModelKvOverrideNative nativeOverride = new LlamaModelKvOverrideNative();
        nativeOverride.tag = type.getValue();
        nativeOverride.setKey(key);
        nativeOverride.value = new LlamaModelKvOverrideNative.ValueUnion();
        switch (type) {
            case LLAMA_KV_OVERRIDE_TYPE_INT:
                nativeOverride.value.setType(long.class);
                nativeOverride.value.val_i64 = longValue;
                break;
            case LLAMA_KV_OVERRIDE_TYPE_FLOAT:
                nativeOverride.value.setType(double.class);
                nativeOverride.value.val_f64 = doubleValue;
                break;
            case LLAMA_KV_OVERRIDE_TYPE_BOOL:
                nativeOverride.value.setType(byte.class);
                nativeOverride.value.val_bool = (byte) (booleanValue ? 1 : 0);
                break;
            case LLAMA_KV_OVERRIDE_TYPE_STR:
                nativeOverride.value.setType(byte[].class);
                nativeOverride.value.val_str = new byte[128];
                if (stringValue != null) {
                    byte[] bytes = stringValue.getBytes();
                    int length = Math.min(bytes.length, nativeOverride.value.val_str.length - 1);
                    System.arraycopy(bytes, 0, nativeOverride.value.val_str, 0, length);
                    nativeOverride.value.val_str[length] = 0;
                }
                break;
            default:
                throw new IllegalStateException("Unsupported override type: " + type);
        }
        nativeOverride.value.write();
        nativeOverride.write();
        return nativeOverride;
    }
}


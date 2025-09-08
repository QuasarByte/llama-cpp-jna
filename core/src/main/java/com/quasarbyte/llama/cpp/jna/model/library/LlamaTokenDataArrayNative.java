package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import java.util.Arrays;

/**
 * JNA mapping of {@code llama_token_data_array}.
 */
@Structure.FieldOrder({"data", "size", "selected", "sorted"})
public class LlamaTokenDataArrayNative extends Structure {

    /** pointer to {@link LlamaTokenDataNative} array (mutable) */
    public Pointer data;

    /** number of items in {@link #data} */
    public long size;

    /** index of the selected token (within {@link #data}), or -1 if unset */
    public long selected;

    /** whether the array is sorted (non-zero == true) */
    public byte sorted;

    private transient LlamaTokenDataNative[] cachedEntries;

    public LlamaTokenDataArrayNative() {
        super();
    }

    public LlamaTokenDataArrayNative(Pointer pointer) {
        super(pointer);
        read();
    }

    /**
     * Returns true when the native structure marks the collection as sorted.
     */
    public boolean isSorted() {
        return sorted != 0;
    }

    /**
     * Updates the sorted flag.
     */
    public void setSorted(boolean value) {
        this.sorted = (byte) (value ? 1 : 0);
    }

    /**
     * Provides safe access to the underlying token data entries by reading the native memory.
     */
    public LlamaTokenDataNative[] getEntries() {
        if (data == null || Pointer.NULL.equals(data) || size <= 0) {
            return new LlamaTokenDataNative[0];
        }

        LlamaTokenDataNative template = new LlamaTokenDataNative(data);
        LlamaTokenDataNative[] entries = (LlamaTokenDataNative[]) template.toArray((int) size);
        Arrays.stream(entries).forEach(LlamaTokenDataNative::read);
        cachedEntries = entries;
        return entries;
    }

    /**
     * Stores the provided entries inside the structure (contiguously allocated) and updates pointers accordingly.
     */
    public void setEntries(LlamaTokenDataNative[] entries) {
        if (entries == null || entries.length == 0) {
            this.data = Pointer.NULL;
            this.size = 0;
            this.cachedEntries = null;
            return;
        }

        LlamaTokenDataNative template = new LlamaTokenDataNative();
        LlamaTokenDataNative[] nativeEntries = (LlamaTokenDataNative[]) template.toArray(entries.length);
        for (int i = 0; i < entries.length; i++) {
            nativeEntries[i].id = entries[i].id;
            nativeEntries[i].logit = entries[i].logit;
            nativeEntries[i].p = entries[i].p;
            nativeEntries[i].write();
        }

        this.cachedEntries = nativeEntries;
        this.data = nativeEntries[0].getPointer();
        this.size = entries.length;
    }

    public static class ByReference extends LlamaTokenDataArrayNative implements Structure.ByReference {
        public ByReference() {
            super();
        }

        public ByReference(Pointer pointer) {
            super(pointer);
        }
    }

    public static class ByValue extends LlamaTokenDataArrayNative implements Structure.ByValue {
        public ByValue() {
            super();
        }
    }
}

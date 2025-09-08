package com.quasarbyte.llama.cpp.jna.model.library;

/**
 * Plain Java wrapper for {@link LlamaMemoryManagerNative} to simplify debugging.
 */
public class LlamaMemoryManager {

    private LlamaMemoryManagerNative pointer;

    public LlamaMemoryManager() {
    }

    public LlamaMemoryManager(LlamaMemoryManagerNative pointer) {
        this.pointer = pointer;
    }

    public LlamaMemoryManagerNative getPointer() {
        return pointer;
    }

    public LlamaMemoryManager setPointer(LlamaMemoryManagerNative pointer) {
        this.pointer = pointer;
        return this;
    }

    public boolean isValid() {
        return pointer != null && pointer.getPointer() != null;
    }
}

package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;

/**
 * Wrapper class for a LLaMA LoRA adapter pointer.
 * <p>
 * This class provides a Java wrapper around the native llama_adapter pointer
 * for Low-Rank Adaptation (LoRA) fine-tuning adapters.
 * <p>
 * LoRA adapters allow applying fine-tuned modifications to models without
 * modifying the original model weights directly.
 */
public class LlamaAdapter {

    private final LlamaAdapterNative nativePointer;

    /**
     * Creates a new LlamaAdapter wrapper.
     *
     * @param nativePointer pointer to the native llama_adapter structure
     */
    public LlamaAdapter(LlamaAdapterNative nativePointer) {
        this.nativePointer = nativePointer;
    }

    /**
     * Gets the native pointer to the llama_adapter structure.
     *
     * @return native pointer
     */
    public LlamaAdapterNative getNativePointer() {
        return nativePointer;
    }

    /**
     * Checks if this adapter wrapper is valid (non-null pointer).
     *
     * @return true if the native pointer is not null, false otherwise
     */
    public boolean isValid() {
        return nativePointer != null;
    }

    @Override
    public String toString() {
        return "LlamaAdapter{" +
                "nativePointer=" + nativePointer +
                '}';
    }
}
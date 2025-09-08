package com.quasarbyte.llama.cpp.jna.binding.llama.adapter;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaAdapter;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;
import com.sun.jna.Pointer;

/**
 * Service for managing LoRA adapters and control vectors in LLaMA models.
 * <p>
 * This binding provides methods for loading, applying, and managing
 * Low-Rank Adaptation (LoRA) adapters and control vectors that can
 * modify model behavior without changing the base model weights.
 */
public interface LlamaAdapterBinding {

    /**
     * Load a LoRA adapter from file.
     * <p>
     * Creates a LoRA adapter that can be applied to contexts for fine-tuned behavior.
     * The adapter will be automatically freed when the associated model is deleted.
     *
     * @param model the model to associate the adapter with
     * @param adapterPath path to the LoRA adapter file
     * @return the loaded LoRA adapter, or null on failure
     */
    LlamaAdapter loadLoraAdapter(LlamaModel model, String adapterPath);

    /**
     * Get LoRA adapter metadata value as string by key name.
     *
     * @param adapter the LoRA adapter
     * @param key metadata key name
     * @return the metadata value, or null if not found
     */
    String getAdapterMetadata(LlamaAdapter adapter, String key);

    /**
     * Get the number of metadata key/value pairs in the LoRA adapter.
     *
     * @param adapter the LoRA adapter
     * @return number of metadata pairs
     */
    int getAdapterMetadataCount(LlamaAdapter adapter);

    /**
     * Get LoRA adapter metadata key name by index.
     *
     * @param adapter the LoRA adapter
     * @param index metadata index
     * @return the metadata key name, or null if index is invalid
     */
    String getAdapterMetadataKey(LlamaAdapter adapter, int index);

    /**
     * Get LoRA adapter metadata value as string by index.
     *
     * @param adapter the LoRA adapter
     * @param index metadata index
     * @return the metadata value, or null if index is invalid
     */
    String getAdapterMetadataValue(LlamaAdapter adapter, int index);

    /**
     * Get the number of invocation tokens if the current LoRA is an ALoRA.
     *
     * @param adapter the LoRA adapter
     * @return number of invocation tokens
     */
    long getAloraInvocationTokenCount(LlamaAdapter adapter);

    /**
     * Get the invocation tokens if the current LoRA is an ALoRA.
     *
     * @param adapter the LoRA adapter
     * @return array of invocation token IDs, or null if not ALoRA
     */
    int[] getAloraInvocationTokens(LlamaAdapter adapter);

    /**
     * Add a loaded LoRA adapter to given context.
     * <p>
     * This will not modify the model's weights directly but applies the adapter
     * during inference.
     *
     * @param context the context to apply the adapter to
     * @param adapter the LoRA adapter to apply
     * @param scale scaling factor for the adapter (typically 1.0)
     */
    void setLoraAdapter(LlamaContext context, LlamaAdapter adapter, float scale);

    /**
     * Remove a specific LoRA adapter from given context.
     *
     * @param context the context to remove the adapter from
     * @param adapter the LoRA adapter to remove
     */
    void removeLoraAdapter(LlamaContext context, LlamaAdapter adapter);

    /**
     * Remove all LoRA adapters from given context.
     *
     * @param context the context to clear adapters from
     */
    void clearLoraAdapters(LlamaContext context);

    /**
     * Apply a loaded control vector to a context, or clear the currently loaded vector.
     * <p>
     * Control vectors allow steering model behavior by applying directional
     * modifications to the model's internal representations.
     *
     * @param context the context to apply the control vector to
     * @param data pointer to control vector data (float array), or null to clear
     * @param length length of the data buffer
     * @param embeddingSize embedding dimension (size of single layer control)
     * @param startLayer start layer index (inclusive)
     * @param endLayer end layer index (inclusive)
     */
    void applyControlVector(LlamaContext context, Pointer data, long length,
                              int embeddingSize, int startLayer, int endLayer);

    /**
     * Clear the currently applied control vector from a context.
     *
     * @param context the context to clear the control vector from
     */
    void clearControlVector(LlamaContext context);

    /**
     * Manually freeSampler a LoRA adapter.
     * <p>
     * Note: loaded adapters will be automatically freed when the associated model is deleted.
     *
     * @param adapter the LoRA adapter to freeSampler
     */
    void freeAdapter(LlamaAdapter adapter);
}
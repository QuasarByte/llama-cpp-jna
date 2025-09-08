package com.quasarbyte.llama.cpp.jna.binding.llama.processing;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaBatch;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;

/**
 * Service for performing inference operations on LLaMA models.
 * <p>
 * This binding handles the core inference operations including encoding
 * (for embeddings) and decoding (for text generation) using batch processing.
 */
public interface LlamaProcessingBinding {

    /**
     * Encode input batch for embeddings generation.
     * <p>
     * Processes the input batch to generate embeddings/representations.
     * This is typically used for sentence embeddings, similarity search,
     * or other representation learning tasks.
     *
     * @param context the context to process with
     * @param batch the input batch containing tokens or embeddings
     * @return 0 on success, positive on warning, negative on error
     */
    int encodeBatch(LlamaContext context, LlamaBatch batch);

    /**
     * Decode input batch for text generation.
     * <p>
     * Processes the input batch and updates the model's internal state
     * for text generation. After calling this, use result access functions
     * to get the output probabilities/logits.
     *
     * @param context the context to process with
     * @param batch the input batch containing tokens
     * @return 0 on success, positive on warning, negative on error
     */
    int decodeBatch(LlamaContext context, LlamaBatch batch);

    /**
     * Check if the last processing operation was successful.
     *
     * @param result the result code from encodeBatch/decodeBatch
     * @return true if successful, false on error
     */
    boolean isSuccess(int result);

    /**
     * Check if the last processing operation produced a warning.
     *
     * @param result the result code from encodeBatch/decodeBatch
     * @return true if there was a warning, false otherwise
     */
    boolean hasWarning(int result);

    /**
     * Check if the last processing operation failed.
     *
     * @param result the result code from encodeBatch/decodeBatch
     * @return true if there was an error, false otherwise
     */
    boolean hasError(int result);

    /**
     * Get a human-readable description of the processing result.
     *
     * @param result the result code from encodeBatch/decodeBatch
     * @return description of the result
     */
    String getResultDescription(int result);
}
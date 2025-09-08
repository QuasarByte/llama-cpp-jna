package com.quasarbyte.llama.cpp.jna.binding.llama.result;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.sun.jna.Pointer;

import java.util.Optional;

/**
 * Service for accessing inference results from LLaMA models.
 * <p>
 * This binding provides methods to retrieve logits (for text generation)
 * and embeddings (for representation learning) after processing operations.
 */
public interface LlamaResultBinding {

    /**
     * Get output logits for the last token in the last sequence.
     * <p>
     * Returns the probability distribution over the vocabulary for the
     * last processed token, used for next-token prediction.
     *
     * @param context the context to get logits from
     * @return array of logits (probabilities), or null if no logits available
     */
    Optional<float[]> getLogits(LlamaContext context);

    /**
     * Get output logits for a specific token position.
     * <p>
     * Returns the probability distribution for the token at the specified position
     * in the last processed batch.
     *
     * @param context the context to get logits from
     * @param tokenIndex the token position index
     * @return array of logits for the specified position, or null if invalid index
     */
    Optional<float[]> getLogitsAt(LlamaContext context, int tokenIndex);

    /**
     * Get embeddings for the last token in the last sequence.
     * <p>
     * Returns the embedding/representation vector for the last processed token,
     * used for similarity search, clustering, or other representation tasks.
     *
     * @param context the context to get embeddings from
     * @return array of embeddings, or null if no embeddings available
     */
    Optional<float[]> getEmbeddings(LlamaContext context);

    /**
     * Get embeddings for a specific token position.
     * <p>
     * Returns the embedding vector for the token at the specified position
     * in the last processed batch.
     *
     * @param context the context to get embeddings from
     * @param tokenIndex the token position index
     * @return array of embeddings for the specified position, or null if invalid index
     */
    Optional<float[]> getEmbeddingsAt(LlamaContext context, int tokenIndex);

    /**
     * Get embeddings for a specific sequence.
     * <p>
     * Returns the embedding vector for the specified sequence ID,
     * useful when processing multiple sequences in parallel.
     *
     * @param context the context to get embeddings from
     * @param sequenceId the sequence ID
     * @return array of embeddings for the specified sequence, or null if invalid sequence
     */
    Optional<float[]> getEmbeddingsForSequence(LlamaContext context, int sequenceId);

    /**
     * Get the vocabulary size for interpreting logits.
     * <p>
     * Returns the size of the vocabulary, which is the length of the logits array.
     *
     * @param context the context to get vocabulary size from
     * @return vocabulary size
     */
    int getVocabularySize(LlamaContext context);

    /**
     * Get the embedding dimension for interpreting embeddings.
     * <p>
     * Returns the dimension of embedding vectors.
     *
     * @param context the context to get embedding dimension from
     * @return embedding dimension
     */
    int getEmbeddingDimension(LlamaContext context);

    /**
     * Check if logits are available for the context.
     *
     * @param context the context to check
     * @return true if logits are available, false otherwise
     */
    boolean hasLogits(LlamaContext context);

    /**
     * Check if embeddings are available for the context.
     *
     * @param context the context to check
     * @return true if embeddings are available, false otherwise
     */
    boolean hasEmbeddings(LlamaContext context);

    /**
     * Enable or disable embeddings computation for the context.
     * <p>
     * When enabled, the model will compute embeddings during inference,
     * which can then be retrieved using the getEmbeddings* methods.
     * Disabling embeddings can improve performance when only logits are needed.
     *
     * @param context the context to configure
     * @param embeddings true to enable embeddings computation, false to disable
     */
    void setEmbeddingsEnabled(LlamaContext context, boolean embeddings);

    // Raw Pointer Access Methods (for advanced use cases)
    
    /**
     * Get raw pointer to output logits for the last token in the last sequence.
     * <p>
     * WARNING: This returns a raw pointer that must be handled carefully.
     * Prefer using the typed array methods when possible.
     *
     * @param context the context to get logits from
     * @return raw pointer to logits array, or null if no logits available
     */
    Optional<Pointer> getLogitsRaw(LlamaContext context);

    /**
     * Get raw pointer to output logits for a specific token position.
     *
     * @param context the context to get logits from
     * @param tokenIndex the token position index
     * @return raw pointer to logits array for the specified position
     */
    Optional<Pointer> getLogitsRawAt(LlamaContext context, int tokenIndex);

    /**
     * Get raw pointer to embeddings for the last token in the last sequence.
     *
     * @param context the context to get embeddings from
     * @return raw pointer to embeddings array, or null if no embeddings available
     */
    Optional<Pointer> getEmbeddingsRaw(LlamaContext context);

    /**
     * Get raw pointer to embeddings for a specific token position.
     *
     * @param context the context to get embeddings from
     * @param tokenIndex the token position index
     * @return raw pointer to embeddings array for the specified position
     */
    Optional<Pointer> getEmbeddingsRawAt(LlamaContext context, int tokenIndex);

    /**
     * Get raw pointer to embeddings for a specific sequence.
     *
     * @param context the context to get embeddings from
     * @param sequenceId the sequence ID
     * @return raw pointer to embeddings array for the specified sequence
     */
    Optional<Pointer> getEmbeddingsRawForSequence(LlamaContext context, int sequenceId);
}
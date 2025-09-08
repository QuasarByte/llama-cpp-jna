package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

/**
 * Native structure for llama batch processing.
 * <p>
 * This corresponds to the {@code llama_batch} structure from llama.h.
 * Used as input data for {@code llama_encode} and {@code llama_decode} functions.
 * <p>
 * A llama_batch object can contain input about one or many sequences.
 * The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens.
 * <p>
 * <strong>Usage Modes:</strong>
 * <ul>
 *   <li><strong>Token Mode:</strong> Use {@code token} array, set {@code embd} to null</li>
 *   <li><strong>Embedding Mode:</strong> Use {@code embd} array, set {@code token} to null</li>
 * </ul>
 *
 * <strong>Array Semantics:</strong>
 * <ul>
 *   <li>{@code pos}: If null, positions are tracked automatically</li>
 *   <li>{@code seq_id}: If null, all tokens belong to sequence 0</li>
 *   <li>{@code logits}: If null, output behavior depends on embeddings setting</li>
 * </ul>
 */
@Structure.FieldOrder({"n_tokens", "token", "embd", "pos", "n_seq_id", "seq_id", "logits"})
public class LlamaBatchNative extends Structure implements Structure.ByValue {
    /**
     * Number of tokens in this batch
     */
    public int n_tokens;

    /**
     * Token IDs of the input (llama_token*).
     * <p>
     * Used when embd is NULL. Array must have size of n_tokens.
     * Each element is a token ID from the model's vocabulary.
     */
    public Pointer token;

    /**
     * Token embeddings (float*).
     * <p>
     * Used when token is NULL. Array of float vectors of size n_embd.
     * Each token is represented as a dense vector instead of a discrete ID.
     * Total array size must be n_tokens * n_embd floats.
     */
    public Pointer embd;

    /**
     * Positions of the respective tokens in the sequence (llama_pos*).
     * <p>
     * If set to NULL, the token positions will be tracked automatically
     * by llama_encode/llama_decode. Array must have size of n_tokens.
     * Position indicates where in the sequence each token appears.
     */
    public Pointer pos;

    /**
     * Number of sequence IDs for each token (int32_t*).
     * <p>
     * Array must have size of n_tokens. Specifies how many sequences
     * each token belongs to (for multi-sequence processing).
     */
    public Pointer n_seq_id;

    /**
     * Sequence IDs for each token (llama_seq_id**).
     * <p>
     * If set to NULL, the sequence ID will be assumed to be 0 for all tokens.
     * This is a pointer to an array of pointers, where each element points to
     * an array of sequence IDs that the corresponding token belongs to.
     * Allows tokens to belong to multiple sequences simultaneously.
     */
    public Pointer seq_id;

    /**
     * Logits output control flags (int8_t*).
     * <p>
     * Array must have size of n_tokens. If zero, the logits (and/or embeddings)
     * for the respective token will not be output.
     * <p>
     * <strong>If set to NULL:</strong>
     * <ul>
     *   <li><strong>If embeddings enabled:</strong> all tokens are output</li>
     *   <li><strong>If embeddings disabled:</strong> only the last token is output</li>
     * </ul>
     */
    public Pointer logits;

    /**
     * Defines the order of fields in the native structure.
     * <p>
     * This order must match exactly with the C structure definition
     * to ensure proper memory layout and JNA marshalling.
     */
    @Override
    protected java.util.List<String> getFieldOrder() {
        return java.util.Arrays.asList("n_tokens", "token", "embd", "pos", "n_seq_id", "seq_id", "logits");
    }

    /**
     * Validates the batch structure to prevent memory access violations.
     * Should be called after native initialization but before using the batch.
     *
     * @throws IllegalStateException if the batch is in an invalid state
     */
    public void validate() {
        if (n_tokens <= 0) {
            throw new IllegalStateException("Invalid batch: n_tokens must be positive, got: " + n_tokens);
        }

        // For token-based batches (most common case), token pointer must be valid
        if (token != null && !token.equals(Pointer.NULL)) {
            // Token mode: embd should be null, token should be valid
            return; // Valid token-based batch
        }

        // For embedding-based batches, embd pointer must be valid
        if (embd != null && !embd.equals(Pointer.NULL)) {
            // Embedding mode: token should be null, embd should be valid
            return; // Valid embedding-based batch
        }

        // Neither token nor embd is valid - this is an error
        throw new IllegalStateException("Invalid batch: both token and embd pointers are null or invalid");
    }

}

package com.quasarbyte.llama.cpp.jna.model.library.llama;

/**
 * Enumeration of attention types supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_attention_type} enumeration from llama.h.
 * These represent different attention mechanisms used in transformer models.
 */
public enum LlamaAttentionType {
    
    /** Unspecified attention type */
    LLAMA_ATTENTION_TYPE_UNSPECIFIED(-1),
    
    /** Causal attention - tokens can only attend to previous positions (autoregressive) */
    LLAMA_ATTENTION_TYPE_CAUSAL(0),
    
    /** Non-causal attention - tokens can attend to all positions (bidirectional) */
    LLAMA_ATTENTION_TYPE_NON_CAUSAL(1);
    
    private final int value;
    
    LlamaAttentionType(int value) {
        this.value = value;
    }
    
    /**
     * Gets the integer value corresponding to the native enum value.
     * 
     * @return the integer value
     */
    public int getValue() {
        return value;
    }
    
    /**
     * Returns the enum constant for the given integer value.
     * 
     * @param value the integer value from the native enum
     * @return the corresponding enum constant
     * @throws IllegalArgumentException if no enum constant matches the value
     */
    public static LlamaAttentionType fromValue(int value) {
        for (LlamaAttentionType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown attention type value: " + value);
    }
    
    /**
     * Checks if this attention type is specified (not unspecified).
     * 
     * @return true if the attention type is specified
     */
    public boolean isSpecified() {
        return this != LLAMA_ATTENTION_TYPE_UNSPECIFIED;
    }
    
    /**
     * Checks if this is causal (autoregressive) attention.
     * 
     * @return true if this is causal attention
     */
    public boolean isCausal() {
        return this == LLAMA_ATTENTION_TYPE_CAUSAL;
    }
    
    /**
     * Checks if this is non-causal (bidirectional) attention.
     * 
     * @return true if this is non-causal attention
     */
    public boolean isNonCausal() {
        return this == LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    }
    
    /**
     * Checks if this attention type is suitable for generative tasks.
     * Causal attention is typically used for text generation.
     * 
     * @return true if suitable for generative tasks
     */
    public boolean isGenerative() {
        return this == LLAMA_ATTENTION_TYPE_CAUSAL;
    }
    
    /**
     * Checks if this attention type is suitable for embedding/encoding tasks.
     * Non-causal attention is typically used for embeddings and classification.
     * 
     * @return true if suitable for embedding tasks
     */
    public boolean isEmbedding() {
        return this == LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    }
}
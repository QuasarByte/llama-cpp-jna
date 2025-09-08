package com.quasarbyte.llama.cpp.jna.model.library.llama;

/**
 * Enumeration of pooling types supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_pooling_type} enumeration from llama.h.
 * These represent different methods for pooling embeddings across sequence positions.
 */
public enum LlamaPoolingType {
    
    /** Unspecified pooling type */
    LLAMA_POOLING_TYPE_UNSPECIFIED(-1),
    
    /** No pooling - return embeddings for all positions */
    LLAMA_POOLING_TYPE_NONE(0),
    
    /** Mean pooling - average embeddings across all positions */
    LLAMA_POOLING_TYPE_MEAN(1),
    
    /** CLS pooling - use embedding from CLS token position */
    LLAMA_POOLING_TYPE_CLS(2),
    
    /** Last pooling - use embedding from last position */
    LLAMA_POOLING_TYPE_LAST(3),
    
    /** Rank pooling - used by reranking models to attach classification head to the graph */
    LLAMA_POOLING_TYPE_RANK(4);
    
    private final int value;
    
    LlamaPoolingType(int value) {
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
    public static LlamaPoolingType fromValue(int value) {
        for (LlamaPoolingType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown pooling type value: " + value);
    }
    
    /**
     * Checks if this pooling type is specified (not unspecified).
     * 
     * @return true if the pooling type is specified
     */
    public boolean isSpecified() {
        return this != LLAMA_POOLING_TYPE_UNSPECIFIED;
    }
    
    /**
     * Checks if this pooling type reduces the sequence to a single embedding.
     * 
     * @return true if this pooling type produces a single embedding per sequence
     */
    public boolean reduceToSingle() {
        return this == LLAMA_POOLING_TYPE_MEAN || 
               this == LLAMA_POOLING_TYPE_CLS || 
               this == LLAMA_POOLING_TYPE_LAST ||
               this == LLAMA_POOLING_TYPE_RANK;
    }
    
    /**
     * Checks if this pooling type is used for classification tasks.
     * 
     * @return true if this pooling type is designed for classification
     */
    public boolean isForClassification() {
        return this == LLAMA_POOLING_TYPE_CLS || this == LLAMA_POOLING_TYPE_RANK;
    }
    
    /**
     * Checks if this pooling type requires a special token (like CLS).
     * 
     * @return true if this pooling type requires a special token
     */
    public boolean requiresSpecialToken() {
        return this == LLAMA_POOLING_TYPE_CLS;
    }
}
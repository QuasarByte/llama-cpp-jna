package com.quasarbyte.llama.cpp.jna.model.library.llama;

/**
 * Enumeration of RoPE (Rotary Position Embedding) types supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_rope_type} enumeration from llama.h
 */
public enum LlamaRopeType {
    
    /** No RoPE type */
    LLAMA_ROPE_TYPE_NONE(-1),
    
    /** Normal RoPE type */
    LLAMA_ROPE_TYPE_NORM(0),
    
    /** GPT-NeoX style RoPE (GGML_ROPE_TYPE_NEOX = 2) */
    LLAMA_ROPE_TYPE_NEOX(2),
    
    /** Multi-modal RoPE (GGML_ROPE_TYPE_MROPE = 8) */
    LLAMA_ROPE_TYPE_MROPE(8),
    
    /** Vision RoPE (GGML_ROPE_TYPE_VISION = 24) */
    LLAMA_ROPE_TYPE_VISION(24);
    
    private final int value;
    
    LlamaRopeType(int value) {
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
    public static LlamaRopeType fromValue(int value) {
        for (LlamaRopeType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown rope type value: " + value);
    }
}
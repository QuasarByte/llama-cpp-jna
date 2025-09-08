package com.quasarbyte.llama.cpp.jna.model.library.llama;

/**
 * Enumeration of RoPE (Rotary Position Embedding) scaling types supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_rope_scaling_type} enumeration from llama.h.
 * These represent different methods for scaling RoPE to handle longer sequences.
 */
public enum LlamaRopeScalingType {
    
    /** Unspecified RoPE scaling type */
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED(-1),
    
    /** No RoPE scaling */
    LLAMA_ROPE_SCALING_TYPE_NONE(0),
    
    /** Linear RoPE scaling */
    LLAMA_ROPE_SCALING_TYPE_LINEAR(1),
    
    /** YaRN (Yet another RoPE extensioN) scaling */
    LLAMA_ROPE_SCALING_TYPE_YARN(2),
    
    /** LongRoPE scaling */
    LLAMA_ROPE_SCALING_TYPE_LONGROPE(3);
    
    /** Maximum value constant (equals LLAMA_ROPE_SCALING_TYPE_LONGROPE) */
    public static final LlamaRopeScalingType LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = LLAMA_ROPE_SCALING_TYPE_LONGROPE;
    
    private final int value;
    
    LlamaRopeScalingType(int value) {
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
    public static LlamaRopeScalingType fromValue(int value) {
        for (LlamaRopeScalingType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown rope scaling type value: " + value);
    }
    
    /**
     * Checks if this scaling type is specified (not unspecified).
     * 
     * @return true if the scaling type is specified
     */
    public boolean isSpecified() {
        return this != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED;
    }
    
    /**
     * Checks if this scaling type involves actual scaling (not NONE or UNSPECIFIED).
     * 
     * @return true if this scaling type applies scaling
     */
    public boolean hasScaling() {
        return this != LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED && this != LLAMA_ROPE_SCALING_TYPE_NONE;
    }
    
    /**
     * Gets the maximum valid scaling type value.
     * 
     * @return the maximum scaling type
     */
    public static LlamaRopeScalingType getMaxValue() {
        return LLAMA_ROPE_SCALING_TYPE_MAX_VALUE;
    }
}
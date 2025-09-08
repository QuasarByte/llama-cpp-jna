package com.quasarbyte.llama.cpp.jna.model.library.llama;

/**
 * Enumeration of Flash Attention types supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_flash_attn_type} enumeration from llama.h.
 * Flash Attention is an optimized attention mechanism that reduces memory usage
 * and improves performance for long sequences.
 */
public enum LlamaFlashAttnType {
    
    /** Automatic Flash Attention - let the system decide based on hardware/model */
    LLAMA_FLASH_ATTN_TYPE_AUTO(-1),
    
    /** Flash Attention disabled - use standard attention implementation */
    LLAMA_FLASH_ATTN_TYPE_DISABLED(0),
    
    /** Flash Attention enabled - use optimized Flash Attention implementation */
    LLAMA_FLASH_ATTN_TYPE_ENABLED(1);
    
    private final int value;
    
    LlamaFlashAttnType(int value) {
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
    public static LlamaFlashAttnType fromValue(int value) {
        for (LlamaFlashAttnType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown flash attention type value: " + value);
    }
    
    /**
     * Checks if Flash Attention is explicitly enabled.
     * 
     * @return true if Flash Attention is explicitly enabled
     */
    public boolean isEnabled() {
        return this == LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    
    /**
     * Checks if Flash Attention is explicitly disabled.
     * 
     * @return true if Flash Attention is explicitly disabled
     */
    public boolean isDisabled() {
        return this == LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    
    /**
     * Checks if Flash Attention is set to automatic mode.
     * In auto mode, the system will decide whether to use Flash Attention
     * based on hardware capabilities and model requirements.
     * 
     * @return true if Flash Attention is in automatic mode
     */
    public boolean isAuto() {
        return this == LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
    
    /**
     * Checks if Flash Attention could potentially be active.
     * This includes both explicitly enabled and auto modes.
     * 
     * @return true if Flash Attention could be active
     */
    public boolean couldBeActive() {
        return this == LLAMA_FLASH_ATTN_TYPE_ENABLED || this == LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
}
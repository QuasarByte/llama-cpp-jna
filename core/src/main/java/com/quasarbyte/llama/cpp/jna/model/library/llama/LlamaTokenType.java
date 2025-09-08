package com.quasarbyte.llama.cpp.jna.model.library.llama;

/**
 * Enumeration of token types supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_token_type} enumeration from llama.h
 * <p>
 * <strong>Note:</strong> This enum may be removed in the future, as it is required until 
 * per token attributes are available from GGUF file.
 * 
 * @deprecated This enum may be removed once per token attributes are available from GGUF file
 */
@Deprecated
public enum LlamaTokenType {
    
    /** Undefined token type */
    LLAMA_TOKEN_TYPE_UNDEFINED(0),
    
    /** Normal token type */
    LLAMA_TOKEN_TYPE_NORMAL(1),
    
    /** Unknown token type */
    LLAMA_TOKEN_TYPE_UNKNOWN(2),
    
    /** Control token type */
    LLAMA_TOKEN_TYPE_CONTROL(3),
    
    /** User defined token type */
    LLAMA_TOKEN_TYPE_USER_DEFINED(4),
    
    /** Unused token type */
    LLAMA_TOKEN_TYPE_UNUSED(5),
    
    /** Byte token type */
    LLAMA_TOKEN_TYPE_BYTE(6);
    
    private final int value;
    
    LlamaTokenType(int value) {
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
    public static LlamaTokenType fromValue(int value) {
        for (LlamaTokenType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown token type value: " + value);
    }
}
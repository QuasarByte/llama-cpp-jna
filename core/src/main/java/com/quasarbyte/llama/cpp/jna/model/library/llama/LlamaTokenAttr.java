package com.quasarbyte.llama.cpp.jna.model.library.llama;

import java.util.EnumSet;
import java.util.Set;

/**
 * Enumeration of token attributes supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_token_attr} enumeration from llama.h.
 * These are bit flags that can be combined to represent multiple attributes.
 */
public enum LlamaTokenAttr {
    
    /** Undefined token attribute */
    LLAMA_TOKEN_ATTR_UNDEFINED(0),
    
    /** Unknown token attribute (1 << 0 = 1) */
    LLAMA_TOKEN_ATTR_UNKNOWN(1 << 0),
    
    /** Unused token attribute (1 << 1 = 2) */
    LLAMA_TOKEN_ATTR_UNUSED(1 << 1),
    
    /** Normal token attribute (1 << 2 = 4) */
    LLAMA_TOKEN_ATTR_NORMAL(1 << 2),
    
    /** Control token attribute (1 << 3 = 8) - SPECIAL? */
    LLAMA_TOKEN_ATTR_CONTROL(1 << 3),
    
    /** User defined token attribute (1 << 4 = 16) */
    LLAMA_TOKEN_ATTR_USER_DEFINED(1 << 4),
    
    /** Byte token attribute (1 << 5 = 32) */
    LLAMA_TOKEN_ATTR_BYTE(1 << 5),
    
    /** Normalized token attribute (1 << 6 = 64) */
    LLAMA_TOKEN_ATTR_NORMALIZED(1 << 6),
    
    /** Left strip token attribute (1 << 7 = 128) */
    LLAMA_TOKEN_ATTR_LSTRIP(1 << 7),
    
    /** Right strip token attribute (1 << 8 = 256) */
    LLAMA_TOKEN_ATTR_RSTRIP(1 << 8),
    
    /** Single word token attribute (1 << 9 = 512) */
    LLAMA_TOKEN_ATTR_SINGLE_WORD(1 << 9);
    
    private final int value;
    
    LlamaTokenAttr(int value) {
        this.value = value;
    }
    
    /**
     * Gets the integer value corresponding to the native enum value.
     * 
     * @return the integer value (bit flag)
     */
    public int getValue() {
        return value;
    }
    
    /**
     * Returns the enum constant for the given integer value.
     * Note: This only works for single bit flags, not combined values.
     * 
     * @param value the integer value from the native enum
     * @return the corresponding enum constant
     * @throws IllegalArgumentException if no enum constant matches the value
     */
    public static LlamaTokenAttr fromValue(int value) {
        for (LlamaTokenAttr attr : values()) {
            if (attr.value == value) {
                return attr;
            }
        }
        throw new IllegalArgumentException("Unknown token attribute value: " + value);
    }
    
    /**
     * Converts a combined bit mask to a set of enum constants.
     * 
     * @param mask the combined bit mask
     * @return a set of enum constants that are set in the mask
     */
    public static Set<LlamaTokenAttr> fromMask(int mask) {
        Set<LlamaTokenAttr> result = EnumSet.noneOf(LlamaTokenAttr.class);
        for (LlamaTokenAttr attr : values()) {
            if (attr.value != 0 && (mask & attr.value) != 0) {
                result.add(attr);
            }
        }
        return result;
    }
    
    /**
     * Converts a set of enum constants to a combined bit mask.
     * 
     * @param attributes the set of attributes
     * @return the combined bit mask
     */
    public static int toMask(Set<LlamaTokenAttr> attributes) {
        int mask = 0;
        for (LlamaTokenAttr attr : attributes) {
            mask |= attr.value;
        }
        return mask;
    }
    
    /**
     * Checks if a specific attribute is set in the given mask.
     * 
     * @param mask the bit mask to check
     * @param attribute the attribute to check for
     * @return true if the attribute is set in the mask
     */
    public static boolean hasAttribute(int mask, LlamaTokenAttr attribute) {
        return (mask & attribute.value) != 0;
    }
}
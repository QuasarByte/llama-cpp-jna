package com.quasarbyte.llama.cpp.jna.model.library.llama;

/**
 * Enumeration of key-value override types for model metadata supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_model_kv_override_type} enumeration from llama.h.
 * These types specify the data type for overriding model metadata key-value pairs.
 */
public enum LlamaModelKvOverrideType {
    
    /** Integer type override */
    LLAMA_KV_OVERRIDE_TYPE_INT(0),
    
    /** Float type override */
    LLAMA_KV_OVERRIDE_TYPE_FLOAT(1),
    
    /** Boolean type override */
    LLAMA_KV_OVERRIDE_TYPE_BOOL(2),
    
    /** String type override */
    LLAMA_KV_OVERRIDE_TYPE_STR(3);
    
    private final int value;
    
    LlamaModelKvOverrideType(int value) {
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
    public static LlamaModelKvOverrideType fromValue(int value) {
        for (LlamaModelKvOverrideType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown KV override type value: " + value);
    }
    
    /**
     * Checks if this type represents a numeric value.
     * 
     * @return true if this type is numeric (int or float)
     */
    public boolean isNumeric() {
        return this == LLAMA_KV_OVERRIDE_TYPE_INT || this == LLAMA_KV_OVERRIDE_TYPE_FLOAT;
    }
    
    /**
     * Checks if this type represents an integer value.
     * 
     * @return true if this type is integer
     */
    public boolean isInteger() {
        return this == LLAMA_KV_OVERRIDE_TYPE_INT;
    }
    
    /**
     * Checks if this type represents a floating-point value.
     * 
     * @return true if this type is float
     */
    public boolean isFloat() {
        return this == LLAMA_KV_OVERRIDE_TYPE_FLOAT;
    }
    
    /**
     * Checks if this type represents a boolean value.
     * 
     * @return true if this type is boolean
     */
    public boolean isBoolean() {
        return this == LLAMA_KV_OVERRIDE_TYPE_BOOL;
    }
    
    /**
     * Checks if this type represents a string value.
     * 
     * @return true if this type is string
     */
    public boolean isString() {
        return this == LLAMA_KV_OVERRIDE_TYPE_STR;
    }
    
    /**
     * Gets the corresponding Java class for this override type.
     * 
     * @return the Java class that represents this type
     */
    public Class<?> getJavaType() {
        switch (this) {
            case LLAMA_KV_OVERRIDE_TYPE_INT:
                return Integer.class;
            case LLAMA_KV_OVERRIDE_TYPE_FLOAT:
                return Float.class;
            case LLAMA_KV_OVERRIDE_TYPE_BOOL:
                return Boolean.class;
            case LLAMA_KV_OVERRIDE_TYPE_STR:
                return String.class;
            default:
                return Object.class;
        }
    }
    
    /**
     * Gets a string representation of the type name.
     * 
     * @return human-readable type name
     */
    public String getTypeName() {
        switch (this) {
            case LLAMA_KV_OVERRIDE_TYPE_INT:
                return "integer";
            case LLAMA_KV_OVERRIDE_TYPE_FLOAT:
                return "float";
            case LLAMA_KV_OVERRIDE_TYPE_BOOL:
                return "boolean";
            case LLAMA_KV_OVERRIDE_TYPE_STR:
                return "string";
            default:
                return "unknown";
        }
    }
}
package com.quasarbyte.llama.cpp.jna.model.library.llama;

/**
 * Enumeration of model file types (quantization formats) supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_ftype} enumeration from llama.h.
 * These represent different quantization formats for model weights.
 */
public enum LlamaFtype {
    
    /** All weights in F32 format */
    LLAMA_FTYPE_ALL_F32(0),
    
    /** Mostly F16, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_F16(1),
    
    /** Mostly Q4_0, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q4_0(2),
    
    /** Mostly Q4_1, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q4_1(3),
    
    // Note: Values 4, 5, 6 are commented out in original C code (removed support)
    
    /** Mostly Q8_0, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q8_0(7),
    
    /** Mostly Q5_0, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q5_0(8),
    
    /** Mostly Q5_1, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q5_1(9),
    
    /** Mostly Q2_K, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q2_K(10),
    
    /** Mostly Q3_K_S, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q3_K_S(11),
    
    /** Mostly Q3_K_M, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q3_K_M(12),
    
    /** Mostly Q3_K_L, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q3_K_L(13),
    
    /** Mostly Q4_K_S, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q4_K_S(14),
    
    /** Mostly Q4_K_M, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q4_K_M(15),
    
    /** Mostly Q5_K_S, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q5_K_S(16),
    
    /** Mostly Q5_K_M, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q5_K_M(17),
    
    /** Mostly Q6_K, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q6_K(18),
    
    /** Mostly IQ2_XXS, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ2_XXS(19),
    
    /** Mostly IQ2_XS, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ2_XS(20),
    
    /** Mostly Q2_K_S, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_Q2_K_S(21),
    
    /** Mostly IQ3_XS, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ3_XS(22),
    
    /** Mostly IQ3_XXS, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ3_XXS(23),
    
    /** Mostly IQ1_S, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ1_S(24),
    
    /** Mostly IQ4_NL, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ4_NL(25),
    
    /** Mostly IQ3_S, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ3_S(26),
    
    /** Mostly IQ3_M, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ3_M(27),
    
    /** Mostly IQ2_S, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ2_S(28),
    
    /** Mostly IQ2_M, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ2_M(29),
    
    /** Mostly IQ4_XS, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ4_XS(30),
    
    /** Mostly IQ1_M, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_IQ1_M(31),
    
    /** Mostly BF16, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_BF16(32),
    
    // Note: Values 33, 34, 35 are commented out in original C code (removed from gguf files)
    
    /** Mostly TQ1_0, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_TQ1_0(36),
    
    /** Mostly TQ2_0, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_TQ2_0(37),
    
    /** Mostly MXFP4_MOE, except 1d tensors */
    LLAMA_FTYPE_MOSTLY_MXFP4_MOE(38),
    
    /** Not specified in the model file (guessed) */
    LLAMA_FTYPE_GUESSED(1024);
    
    private final int value;
    
    LlamaFtype(int value) {
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
    public static LlamaFtype fromValue(int value) {
        for (LlamaFtype ftype : values()) {
            if (ftype.value == value) {
                return ftype;
            }
        }
        throw new IllegalArgumentException("Unknown ftype value: " + value);
    }
    
    /**
     * Checks if this is a quantized format (not F32 or F16).
     * 
     * @return true if this format uses quantization
     */
    public boolean isQuantized() {
        return this != LLAMA_FTYPE_ALL_F32 && this != LLAMA_FTYPE_MOSTLY_F16 && 
               this != LLAMA_FTYPE_MOSTLY_BF16 && this != LLAMA_FTYPE_GUESSED;
    }
    
    /**
     * Checks if this format affects mostly all tensors except 1d tensors.
     * 
     * @return true if format applies to most tensors except 1d tensors
     */
    public boolean isMostlyFormat() {
        return this != LLAMA_FTYPE_ALL_F32 && this != LLAMA_FTYPE_GUESSED;
    }
}
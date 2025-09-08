package com.quasarbyte.llama.cpp.jna.model.library.llama;

/**
 * Enumeration of vocabulary types supported by llama.cpp.
 * <p>
 * This enum corresponds to the {@code llama_vocab_type} enumeration from llama.h
 */
public enum LlamaVocabType {
    
    /** For models without vocab */
    LLAMA_VOCAB_TYPE_NONE(0),
    
    /** LLaMA tokenizer based on byte-level BPE with byte fallback */
    LLAMA_VOCAB_TYPE_SPM(1),
    
    /** GPT-2 tokenizer based on byte-level BPE */
    LLAMA_VOCAB_TYPE_BPE(2),
    
    /** BERT tokenizer based on WordPiece */
    LLAMA_VOCAB_TYPE_WPM(3),
    
    /** T5 tokenizer based on Unigram */
    LLAMA_VOCAB_TYPE_UGM(4),
    
    /** RWKV tokenizer based on greedy tokenization */
    LLAMA_VOCAB_TYPE_RWKV(5),
    
    /** PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming */
    LLAMA_VOCAB_TYPE_PLAMO2(6);
    
    private final int value;
    
    LlamaVocabType(int value) {
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
    public static LlamaVocabType fromValue(int value) {
        for (LlamaVocabType type : values()) {
            if (type.value == value) {
                return type;
            }
        }
        throw new IllegalArgumentException("Unknown vocab type value: " + value);
    }
}
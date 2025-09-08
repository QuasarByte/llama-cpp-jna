package com.quasarbyte.llama.cpp.jna.model.library;

public class LlamaVocabulary {
    private LlamaVocabularyNative vocabularyPointer;

    public LlamaVocabularyNative getVocabularyPointer() {
        return vocabularyPointer;
    }

    public LlamaVocabulary setVocabularyPointer(LlamaVocabularyNative vocabularyPointer) {
        this.vocabularyPointer = vocabularyPointer;
        return this;
    }
}

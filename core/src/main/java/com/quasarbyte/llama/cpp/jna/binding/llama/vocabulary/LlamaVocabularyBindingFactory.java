package com.quasarbyte.llama.cpp.jna.binding.llama.vocabulary;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaVocabularyBindingFactory {

    public LlamaVocabularyBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaVocabularyBindingImpl(llamaLibrary);
    }
}
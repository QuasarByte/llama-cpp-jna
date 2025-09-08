package com.quasarbyte.llama.cpp.jna.binding.llama.token;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaContext;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaVocabulary;

public interface LlamaTokenBinding {

    int[] tokenize(LlamaModel llamaModel, LlamaVocabulary llamaVocabulary, String text, boolean addBos, boolean special);

    String tokenToPiece(LlamaModel llamaModel, LlamaVocabulary llamaVocabulary, int token, boolean special, int maxLength);

    boolean isEndOfGeneration(LlamaVocabulary llamaVocabulary, int token);

    /**
     * Convert tokens back to text (inverse of tokenize).
     *
     * @param vocabulary     the vocabulary to use
     * @param tokens         array of token IDs
     * @param removeSpecial  if true, remove special tokens from output
     * @param unparseSpecial if true, render special tokens as text
     * @return the detokenized text, or null on error
     */
    String detokenize(LlamaVocabulary vocabulary, int[] tokens, boolean removeSpecial, boolean unparseSpecial, int bytesPerToken);

    int getUsedTokens(LlamaContext context);
}

package com.quasarbyte.llama.cpp.jna.binding.llama.vocabulary;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaVocabulary;

import java.util.Optional;

public interface LlamaVocabularyBinding {

    LlamaVocabulary getVocabulary(LlamaModel model);

    int getVocabularyType(LlamaVocabulary vocabulary);

    int getVocabularyTokenCount(LlamaVocabulary vocabulary);

    String getTokenText(LlamaVocabulary vocabulary, int token);

    float getTokenScore(LlamaVocabulary vocabulary, int token);

    int getTokenAttribute(LlamaVocabulary vocabulary, int token);

    boolean isEndOfGeneration(LlamaVocabulary vocabulary, int token);

    boolean isControl(LlamaVocabulary vocabulary, int token);

    boolean getAddBos(LlamaVocabulary vocabulary);

    boolean getAddEos(LlamaVocabulary vocabulary);

    boolean getAddSep(LlamaVocabulary vocabulary);

    /**
     * Gets the Beginning of Sequence (BOS) token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return BOS token ID if available, empty otherwise
     */
    Optional<Integer> getBosToken(LlamaVocabulary vocabulary);

    /**
     * Gets the End of Sequence (EOS) token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return EOS token ID if available, empty otherwise
     */
    Optional<Integer> getEosToken(LlamaVocabulary vocabulary);

    /**
     * Gets the End of Turn (EOT) token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return EOT token ID if available, empty otherwise
     */
    Optional<Integer> getEotToken(LlamaVocabulary vocabulary);

    /**
     * Gets the Separator (SEP) token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return SEP token ID if available, empty otherwise
     */
    Optional<Integer> getSepToken(LlamaVocabulary vocabulary);

    /**
     * Gets the Newline (NL) token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return NL token ID if available, empty otherwise
     */
    Optional<Integer> getNlToken(LlamaVocabulary vocabulary);

    /**
     * Gets the Padding (PAD) token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return PAD token ID if available, empty otherwise
     */
    Optional<Integer> getPadToken(LlamaVocabulary vocabulary);

    /**
     * Gets the Mask token ID (for masked language modeling).
     *
     * @param vocabulary the vocabulary to query
     * @return MASK token ID if available, empty otherwise
     */
    Optional<Integer> getMaskToken(LlamaVocabulary vocabulary);

    /**
     * Gets the Fill-in-the-Middle (FIM) prefix token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return FIM prefix token ID if available, empty otherwise
     */
    Optional<Integer> getFimPreToken(LlamaVocabulary vocabulary);

    /**
     * Gets the Fill-in-the-Middle (FIM) suffix token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return FIM suffix token ID if available, empty otherwise
     */
    Optional<Integer> getFimSufToken(LlamaVocabulary vocabulary);

    /**
     * Gets the Fill-in-the-Middle (FIM) middle token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return FIM middle token ID if available, empty otherwise
     */
    Optional<Integer> getFimMidToken(LlamaVocabulary vocabulary);

    /**
     * Gets the Fill-in-the-Middle (FIM) pad token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return FIM pad token ID if available, empty otherwise
     */
    Optional<Integer> getFimPadToken(LlamaVocabulary vocabulary);

    /**
     * Gets the Fill-in-the-Middle (FIM) repository token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return FIM repository token ID if available, empty otherwise
     */
    Optional<Integer> getFimRepToken(LlamaVocabulary vocabulary);

    /**
     * Gets the Fill-in-the-Middle (FIM) separator token ID.
     *
     * @param vocabulary the vocabulary to query
     * @return FIM separator token ID if available, empty otherwise
     */
    Optional<Integer> getFimSepToken(LlamaVocabulary vocabulary);
}
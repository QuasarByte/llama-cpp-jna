package com.quasarbyte.llama.cpp.jna.binding.llama.vocabulary;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaVocabulary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaVocabularyNative;

import java.util.Objects;
import java.util.Optional;

public class LlamaVocabularyBindingImpl implements LlamaVocabularyBinding {

    private final LlamaLibrary llamaLibrary;

    public LlamaVocabularyBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public LlamaVocabulary getVocabulary(LlamaModel model) {
        return new LlamaVocabulary().setVocabularyPointer(llamaLibrary.llama_model_get_vocab(model.getModelPointer()));
    }

    @Override
    public int getVocabularyType(LlamaVocabulary vocabulary) {
        return llamaLibrary.llama_vocab_type(getVocabularyPointer(vocabulary));
    }

    @Override
    public int getVocabularyTokenCount(LlamaVocabulary vocabulary) {
        return llamaLibrary.llama_vocab_n_tokens(getVocabularyPointer(vocabulary));
    }

    @Override
    public String getTokenText(LlamaVocabulary vocabulary, int token) {
        return llamaLibrary.llama_vocab_get_text(getVocabularyPointer(vocabulary), token);
    }

    @Override
    public float getTokenScore(LlamaVocabulary vocabulary, int token) {
        return llamaLibrary.llama_vocab_get_score(getVocabularyPointer(vocabulary), token);
    }

    @Override
    public int getTokenAttribute(LlamaVocabulary vocabulary, int token) {
        return llamaLibrary.llama_vocab_get_attr(getVocabularyPointer(vocabulary), token);
    }

    @Override
    public boolean isEndOfGeneration(LlamaVocabulary vocabulary, int token) {
        return llamaLibrary.llama_vocab_is_eog(getVocabularyPointer(vocabulary), token);
    }

    @Override
    public boolean isControl(LlamaVocabulary vocabulary, int token) {
        return llamaLibrary.llama_vocab_is_control(getVocabularyPointer(vocabulary), token);
    }

    @Override
    public boolean getAddBos(LlamaVocabulary vocabulary) {
        return llamaLibrary.llama_vocab_get_add_bos(getVocabularyPointer(vocabulary));
    }

    @Override
    public boolean getAddEos(LlamaVocabulary vocabulary) {
        return llamaLibrary.llama_vocab_get_add_eos(getVocabularyPointer(vocabulary));
    }

    @Override
    public boolean getAddSep(LlamaVocabulary vocabulary) {
        return llamaLibrary.llama_vocab_get_add_sep(getVocabularyPointer(vocabulary));
    }

    @Override
    public Optional<Integer> getBosToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_bos(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getEosToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_eos(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getEotToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_eot(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getSepToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_sep(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getNlToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_nl(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getPadToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_pad(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getMaskToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_mask(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getFimPreToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_fim_pre(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getFimSufToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_fim_suf(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getFimMidToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_fim_mid(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getFimPadToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_fim_pad(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getFimRepToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_fim_rep(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    @Override
    public Optional<Integer> getFimSepToken(LlamaVocabulary vocabulary) {
        int token = llamaLibrary.llama_vocab_fim_sep(getVocabularyPointer(vocabulary));
        return token != -1 ? Optional.of(token) : Optional.empty();
    }

    private static LlamaVocabularyNative getVocabularyPointer(LlamaVocabulary vocabulary) {
        Objects.requireNonNull(vocabulary);
        return Objects.requireNonNull(vocabulary.getVocabularyPointer());
    }

}
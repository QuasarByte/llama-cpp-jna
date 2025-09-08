package com.quasarbyte.llama.cpp.jna.binding.llama.token;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.exception.LlamaFunctionCallIntResultException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.*;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReader;
import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Objects;

public class LlamaTokenBindingImpl implements LlamaTokenBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaTokenBindingImpl.class);

    private final LlamaLibrary llamaLibrary;
    private final LlamaStringBufferReader llamaStringBufferReader;

    public LlamaTokenBindingImpl(LlamaLibrary llamaLibrary, LlamaStringBufferReader llamaStringBufferReader) {
        this.llamaLibrary = llamaLibrary;
        this.llamaStringBufferReader = llamaStringBufferReader;
    }

    public int[] tokenize(LlamaModel llamaModel, LlamaVocabulary llamaVocabulary,
                          String text, boolean addBos, boolean parseSpecial) {
        Objects.requireNonNull(llamaModel);
        Objects.requireNonNull(llamaModel.getModelPointer());
        Objects.requireNonNull(llamaVocabulary);
        Objects.requireNonNull(llamaVocabulary.getVocabularyPointer());
        Objects.requireNonNull(text);

        // Convert Java String into UTF-8 bytes (C function expects bytes + length)
        byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);

        // First call: probe how many tokens are needed (passing null for tokens)
        int probe = llamaLibrary.llama_tokenize(
                llamaVocabulary.getVocabularyPointer(),
                utf8,                 // pass as byte[]
                utf8.length,          // length in bytes
                Pointer.NULL,         // no output buffer yet
                0,                    // buffer size = 0
                addBos,
                parseSpecial
        );

        if (probe == Integer.MIN_VALUE) {
            throw new LlamaFunctionCallIntResultException(
                    probe, "Tokenization result size exceeds int32_t limit");
        }

        // If probe >= 0 → result fits in 0 buffer (can happen for empty string)
        // If probe < 0 → negative means how many tokens would have been produced
        int required = (probe >= 0) ? probe : -probe;

        if (required == 0) {
            return new int[0]; // nothing to tokenize
        }

        int[] tokens = new int[required];

        // Allocate native memory for the tokens (int32_t = 4 bytes)
        try (Memory tokenMem = new Memory((long) required * 4L)) {
            int result = llamaLibrary.llama_tokenize(
                    llamaVocabulary.getVocabularyPointer(),
                    utf8,
                    utf8.length,
                    tokenMem, // buffer for output
                    required,
                    addBos,
                    parseSpecial
            );

            if (result == Integer.MIN_VALUE) {
                throw new LlamaFunctionCallIntResultException(
                        result, "Overflow: tokenization result size exceeds int32_t limit");
            }
            if (result < 0) {
                int need = -result;
                throw new LlamaFunctionCallIntResultException(
                        result, "Buffer too small, need at least " + need + " tokens");
            }

            // Copy exactly 'result' ints from native memory into Java array
            tokenMem.read(0, tokens, 0, result);

            // If result < required, shrink the array
            if (result != required) {
                return Arrays.copyOf(tokens, result);
            }

            return tokens;
        } catch (Exception e) {
            throw new LlamaCppJnaException(
                    "Failed to tokenize text, error: " + e.getMessage(), e);
        }
    }

    @Override
    public String tokenToPiece(LlamaModel llamaModel, LlamaVocabulary llamaVocabulary, int token, boolean special, int maxLength) {
        Objects.requireNonNull(llamaModel);
        Objects.requireNonNull(llamaModel.getModelPointer());
        Objects.requireNonNull(llamaVocabulary);
        Objects.requireNonNull(llamaVocabulary.getVocabularyPointer());

        if (maxLength < 0) {
            logger.error("Parameter maxLength can not be less than 0, parameter value: {}", maxLength);
            throw new LlamaCppJnaException(String.format("Parameter maxLength can not be less than 0, parameter value: %d", maxLength));
        }

        byte[] buffer = new byte[maxLength];

        final int length;

        try {
            length = llamaLibrary.llama_token_to_piece(llamaVocabulary.getVocabularyPointer(), token, buffer, buffer.length, 0, special);

            if (length < 0) {
                logger.error("Failed to convert token to Piece, token: {}, special: {}, maxLength: {}, length: {}", token, special, maxLength, length);
                throw new LlamaFunctionCallIntResultException(length, String.format("Failed to convert token to Piece, token: %d, special: %b, maxLength: %d, length: %d", token, special, maxLength, length));
            }

        } catch (Exception e) {
            logger.error("Failed to token to Piece, error: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to convert token to piece, error: %s", e.getMessage()), e);
        }

        return this.llamaStringBufferReader.readString(buffer,length);
    }

    @Override
    public boolean isEndOfGeneration(LlamaVocabulary llamaVocabulary, int token) {
        boolean isEog = llamaLibrary.llama_vocab_is_eog(llamaVocabulary.getVocabularyPointer(), token);
        if (isEog) {
            logger.info("Token {} detected as end-of-generation", token);
        }
        return isEog;
    }

    @Override
    public String detokenize(LlamaVocabulary vocabulary, int[] tokens, boolean removeSpecial, boolean unparseSpecial, int bytesPerToken) {
        Objects.requireNonNull(vocabulary);
        Objects.requireNonNull(vocabulary.getVocabularyPointer());
        Objects.requireNonNull(tokens);

        if (bytesPerToken < 1) {
            logger.error("Parameter bytesPerToken can not be less than 1, parameter value: {}", bytesPerToken);
        }

        // Allocate memory for token array
        // Always clean up memory to prevent heap corruption
        try (Memory tokenMemory = new Memory(tokens.length * 4L)) {
            for (int i = 0; i < tokens.length; i++) {
                tokenMemory.setInt(i * 4L, tokens[i]);
            }

            // First call to get required text length
            int maxTextLen = bytesPerToken*tokens.length;
            byte[] textBuffer = new byte[maxTextLen];

            int length = llamaLibrary.llama_detokenize(
                vocabulary.getVocabularyPointer(),
                tokenMemory,
                tokens.length,
                textBuffer,
                maxTextLen,
                removeSpecial,
                unparseSpecial
            );

            if (length < 0) {
                logger.error("Failed to detokenize tokens, length: {}", length);
                throw new LlamaFunctionCallIntResultException(length, String.format("Failed to detokenize tokens, length: %d", length));
            }

            return this.llamaStringBufferReader.readString(textBuffer, length);

        } catch (Exception e) {
            logger.error("Failed to detokenize tokens, error: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to detokenize tokens, error: %s",  e.getMessage()), e);
        }
    }

    @Override
    public int getUsedTokens(LlamaContext context) {
        Objects.requireNonNull(context);
        Objects.requireNonNull(context.getContextPointer());

        LlamaMemoryManagerNative memoryManager = llamaLibrary.llama_get_memory(context.getContextPointer());
        if (memoryManager == null) {
            throw new LlamaCppJnaException("Invalid context memory manager");
        }

        try {
            // Get the maximum position for sequence 0 (default sequence)
            // This represents the number of tokens processed so far
            int maxPos = llamaLibrary.llama_memory_seq_pos_max(memoryManager, 0);
            // If maxPos is -1, it means the sequence is empty (no tokens processed)
            return maxPos == -1 ? 0 : maxPos + 1;
        } catch (Exception e) {
            logger.error("Failed to get used tokens, error: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to get used tokens, error: %s",  e.getMessage()), e);
        }
    }
}

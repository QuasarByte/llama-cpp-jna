package com.quasarbyte.llama.cpp.jna.binding.llama.chat;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.quasarbyte.llama.cpp.jna.exception.LlamaFunctionCallIntResultException;
import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaChatMessage;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaChatMessageNative;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Implementation of LlamaChatBinding for handling chat template formatting.
 */
public class LlamaChatBindingImpl implements LlamaChatBinding {

    private static final Logger logger = LoggerFactory.getLogger(LlamaChatBindingImpl.class);

    private final LlamaLibrary llamaLibrary;
    private final LlamaStringBufferReader llamaStringBufferReader;

    public LlamaChatBindingImpl(LlamaLibrary llamaLibrary, LlamaStringBufferReader llamaStringBufferReader) {
        this.llamaLibrary = llamaLibrary;
        this.llamaStringBufferReader = llamaStringBufferReader;
    }

    @Override
    public String applyChatTemplate(String template,
                                    List<LlamaChatMessage> messages,
                                    boolean addAssistant,
                                    int maxLength) {
        if (template == null) {
            throw new LlamaCppJnaException("Template cannot be null");
        }

        if (template.trim().isEmpty()) {
            throw new LlamaCppJnaException("Template cannot be blank");
        }

        if (messages == null) {
            throw new LlamaCppJnaException("Messages list cannot be null");
        }

        if (maxLength < 0) {
            logger.error("Max length of messages list cannot be less than 0");
            throw new LlamaCppJnaException("Max length of messages list cannot be less than 0");
        }

        // Create native chat message array
        LlamaChatMessageNative[] nativeMessages = createNativeChatMessageArray(messages);

        try {
            // Prepare output buffer
            byte[] buffer = new byte[maxLength];

            // Call native function with contiguous memory
            int length = llamaLibrary.llama_chat_apply_template(template, nativeMessages, messages.size(), addAssistant, buffer, buffer.length);

            if (length < 0) {
                logger.error("Failed to apply chat template, length: {}, addAssistant: {}, maxLength: {}", length, addAssistant, maxLength);
                throw new LlamaFunctionCallIntResultException(length, String.format("Failed to apply chat template, length: %d, addAssistant: %b, maxLength: %d", length, addAssistant, maxLength));
            }

            return llamaStringBufferReader.readString(buffer, length);

        } catch (Exception e) {
            logger.error("Failed to apply chat template, message: {}", e.getMessage(), e);
            throw new LlamaCppJnaException(String.format("Failed to apply chat template, message: %s", e.getMessage()), e);
        }
    }

    @Override
    public List<String> getBuiltinTemplates(int maxCount) {

        if (maxCount < 0) {
            logger.error("Max count of messages list cannot be less than 0");
            throw new LlamaCppJnaException("Max count of templates list cannot be less than 0");
        }

        String[] output = new String[maxCount];

        try {
            int numberOfTemplates = llamaLibrary.llama_chat_builtin_templates(output, maxCount);

            if (numberOfTemplates < 0) {
                logger.error("Failed to get built-in templates, numberOfTemplates: {}", numberOfTemplates);
                throw new LlamaFunctionCallIntResultException(numberOfTemplates, String.format("Failed to get built-in templates, numberOfTemplates: %d", numberOfTemplates));
            }

            ArrayList<String> result = new ArrayList<>(numberOfTemplates);
            result.addAll(Arrays.asList(output).subList(0, numberOfTemplates));
            return result;
        } catch (Exception e) {
            logger.error("Failed to get built-in templates, error: {}", e.getMessage());
            throw new LlamaCppJnaException(String.format("Failed to get built-in templates, error: %s", e.getMessage()), e);
        }
    }

    @Override
    public LlamaChatMessage createMessage(String role, String content) {

        if (role == null) {
            throw new LlamaCppJnaException("Role cannot be null");
        }

        if (role.trim().isEmpty()) {
            throw new LlamaCppJnaException("Role cannot be blank");
        }

        if (content == null) {
            throw new LlamaCppJnaException("Content cannot be null");
        }

        return new LlamaChatMessage(role.trim(), content);
    }

    @Override
    public LlamaChatMessage createSystemMessage(String content) {
        return createMessage("system", content);
    }

    @Override
    public LlamaChatMessage createUserMessage(String content) {
        return createMessage("user", content);
    }

    @Override
    public LlamaChatMessage createAssistantMessage(String content) {
        return createMessage("assistant", content);
    }

    @Override
    public Optional<String> getChatTemplate(LlamaModel model, String name) {
        Objects.requireNonNull(model);
        Objects.requireNonNull(model.getModelPointer());
        return Optional.ofNullable(llamaLibrary.llama_model_chat_template(model.getModelPointer(), name));
    }

    private static LlamaChatMessageNative[] createNativeChatMessageArray(List<LlamaChatMessage> messages) {
        if (messages.isEmpty()) {
            return new LlamaChatMessageNative[0];
        }

        // Create first element to use as template for toArray
        LlamaChatMessageNative first = new LlamaChatMessageNative();
        first.role = messages.get(0).getRole();
        first.content = messages.get(0).getContent();

        // Create contiguous array using JNA's toArray method
        LlamaChatMessageNative[] nativeMessages = (LlamaChatMessageNative[]) first.toArray(messages.size());

        // Populate the array with message data
        for (int i = 0; i < messages.size(); i++) {
            LlamaChatMessage message = messages.get(i);
            nativeMessages[i].role = message.getRole();
            nativeMessages[i].content = message.getContent();
        }

        return nativeMessages;
    }
}
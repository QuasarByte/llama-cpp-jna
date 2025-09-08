package com.quasarbyte.llama.cpp.jna.binding.llama.chat;

import com.quasarbyte.llama.cpp.jna.model.library.LlamaChatMessage;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaModel;

import java.util.List;
import java.util.Optional;

/**
 * Service for handling chat template formatting and conversation management.
 * <p>
 * This binding provides methods for applying chat templates to format
 * conversation messages for language model inference, following common
 * chat formats like ChatML, Alpaca, etc.
 */
public interface LlamaChatBinding {

    /**
     * Apply a chat template to format conversation messages with maximum length limit.
     * <p>
     * Formats a list of chat messages using the specified template string,
     * producing a single formatted string suitable for model inference.
     *
     * @param template the chat template string
     * @param messages list of chat messages to format
     * @param addAssistant whether to add an assistant message placeholder at the end
     * @param maxLength maximum length of the output string
     * @return formatted chat string (truncated if necessary), or null on error
     */
    String applyChatTemplate(String template, List<LlamaChatMessage> messages, 
                           boolean addAssistant, int maxLength);

    /**
     * Get a list of built-in chat template names.
     * <p>
     * Returns the names of chat templates that are built into the library,
     * such as "chatml", "alpaca", "vicuna", etc.
     *
     * @return list of built-in chat template names, or empty list if none available
     */
    List<String> getBuiltinTemplates(int maxCount);

    /**
     * Create a simple chat message.
     *
     * @param role the role of the message sender (e.g., "system", "user", "assistant")
     * @param content the message content
     * @return a new chat message
     */
    LlamaChatMessage createMessage(String role, String content);

    /**
     * Create a system message.
     *
     * @param content the system message content
     * @return a new system message
     */
    LlamaChatMessage createSystemMessage(String content);

    /**
     * Create a user message.
     *
     * @param content the user message content
     * @return a new user message
     */
    LlamaChatMessage createUserMessage(String content);

    /**
     * Create an assistant message.
     *
     * @param content the assistant message content
     * @return a new assistant message
     */
    LlamaChatMessage createAssistantMessage(String content);

    Optional<String> getChatTemplate(LlamaModel model, String name);

}
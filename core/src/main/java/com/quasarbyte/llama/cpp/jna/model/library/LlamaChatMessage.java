package com.quasarbyte.llama.cpp.jna.model.library;

/**
 * Represents a single message in a chat conversation.
 * <p>
 * This class is used with chat templates to format conversations
 * for language model inference. Messages have a role (system, user, assistant)
 * and content text.
 */
public class LlamaChatMessage {

    private final String role;
    private final String content;

    /**
     * Creates a new chat message.
     *
     * @param role the role of the message sender (e.g., "system", "user", "assistant")
     * @param content the text content of the message
     */
    public LlamaChatMessage(String role, String content) {
        this.role = role;
        this.content = content;
    }

    /**
     * Gets the role of the message sender.
     *
     * @return the role string
     */
    public String getRole() {
        return role;
    }

    /**
     * Gets the content of the message.
     *
     * @return the message content
     */
    public String getContent() {
        return content;
    }

    @Override
    public String toString() {
        return "LlamaChatMessage{" +
                "role='" + role + '\'' +
                ", content='" + content + '\'' +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        LlamaChatMessage that = (LlamaChatMessage) o;

        if (!role.equals(that.role)) return false;
        return content.equals(that.content);
    }

    @Override
    public int hashCode() {
        int result = role.hashCode();
        result = 31 * result + content.hashCode();
        return result;
    }
}
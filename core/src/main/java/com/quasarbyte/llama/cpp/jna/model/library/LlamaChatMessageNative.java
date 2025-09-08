package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import java.util.Arrays;
import java.util.List;

/**
 * JNA mapping for the native llama_chat_message structure.
 * <p>
 * typedef struct llama_chat_message {
 *     const char * role;
 *     const char * content;
 * } llama_chat_message;
 */
@Structure.FieldOrder({"role", "content"})
public class LlamaChatMessageNative extends Structure {

    public String role;
    public String content;

    public LlamaChatMessageNative() {
        super();
    }

    public LlamaChatMessageNative(String role, String content) {
        super();
        this.role = role;
        this.content = content;
    }

    public LlamaChatMessageNative(Pointer pointer) {
        super(pointer);
    }

    public String getRole() {
        return role;
    }

    public LlamaChatMessageNative setRole(String role) {
        this.role = role;
        return this;
    }

    public String getContent() {
        return content;
    }

    public LlamaChatMessageNative setContent(String content) {
        this.content = content;
        return this;
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("role", "content");
    }

    public static class ByReference extends LlamaChatMessageNative implements Structure.ByReference {
        public ByReference() {}
        public ByReference(Pointer pointer) { super(pointer); }
    }

    public static class ByValue extends LlamaChatMessageNative implements Structure.ByValue {
        public ByValue() {}
    }
}
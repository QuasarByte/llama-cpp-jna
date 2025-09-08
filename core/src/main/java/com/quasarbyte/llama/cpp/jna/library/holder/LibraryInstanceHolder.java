package com.quasarbyte.llama.cpp.jna.library.holder;

import com.sun.jna.Library;

import java.util.Map;

public interface LibraryInstanceHolder<T extends Library> {

    String getLibraryName();

    Map<String, Object> getLibraryOptions();

    Class<T> getLibraryClass();

    T getInstance();

    boolean isAvailable();

    Throwable getLoadThrowable();

    String getLoadErrorMessage();
}

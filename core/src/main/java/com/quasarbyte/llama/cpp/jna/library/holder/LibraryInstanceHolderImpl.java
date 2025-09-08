package com.quasarbyte.llama.cpp.jna.library.holder;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import com.sun.jna.Library;
import com.sun.jna.Native;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Thread-safe lazy holder for a JNA library instance.
 * <p>
 * JNA inherits its default string marshalling charset from the runtime environment.
 * On Windows this is typically a legacy code page such as {@code windows-1251}
 * which causes UTF-8 prompts to be mangled when passed to {@code llama.cpp}.
 * <p>
 * To shield library users from having to specify {@code -Djna.encoding=UTF-8}
 * every time they launch an application, this holder ensures UTF-8 is used in two
 * complementary ways:
 * <ul>
 *   <li>During class initialisation the global {@code jna.encoding} system
 *   property is set to UTF-8 if the user has not provided a value.</li>
 *   <li>Every call to {@link com.sun.jna.Native#load(String, Class, java.util.Map)}
 *   receives an explicit {@link com.sun.jna.Library#OPTION_STRING_ENCODING}
 *   entry, so that individual bindings always marshal strings as UTF-8.</li>
 * </ul>
 * These safeguards keep existing command-line workflows intact while guaranteeing
 * correct UTF-8 behaviour for all consumers of the bindings.
 */
public final class LibraryInstanceHolderImpl<T extends Library> implements LibraryInstanceHolder<T> {
    private final String libraryName;
    private final Map<String, Object> libraryOptions;
    private final Class<T> libraryClass;

    static {
        String encoding = System.getProperty("jna.encoding");
        if (encoding == null || encoding.trim().isEmpty()) {
            System.setProperty("jna.encoding", "UTF-8");
        }
    }

    // Volatile fields to ensure visibility across threads
    private volatile T instance;
    private volatile Throwable loadThrowable;

    /**
     * Creates a new holder.
     *
     * @param libraryName   the name of the native library (must not be null/blank)
     * @param libraryOptions optional options for JNA Native.load (may be null)
     * @param libraryClass  the JNA interface class (must not be null)
     */
    public LibraryInstanceHolderImpl(String libraryName, Map<String, Object> libraryOptions, Class<T> libraryClass) {
        if (libraryName == null || libraryName.trim().isEmpty()) {
            throw new IllegalArgumentException("libraryName must not be blank");
        }
        if (libraryClass == null) {
            throw new IllegalArgumentException("libraryClass must not be null");
        }
        this.libraryName = libraryName;
        LinkedHashMap<String, Object> options = new LinkedHashMap<>();
        if (libraryOptions != null) {
            options.putAll(libraryOptions);
        }
        options.putIfAbsent(Library.OPTION_STRING_ENCODING, "UTF-8");
        // Defensive copy + make unmodifiable for safety
        this.libraryOptions = Collections.unmodifiableMap(options);
        this.libraryClass = libraryClass;
    }

    /**
     * Returns the library instance, loading it lazily if necessary.
     * If loading fails, throws a wrapped exception and allows retry on the next call.
     */
    @Override
    public T getInstance() {
        T cur = instance;
        if (cur != null) return cur;

        synchronized (this) {
            if (instance != null) return instance;

            try {
                T loaded = Native.load(libraryName, libraryClass, libraryOptions);
                instance = loaded;
                loadThrowable = null; // clear previous error
                return loaded;
            } catch (Throwable t) {
                loadThrowable = t;
                instance = null; // do not cache failure
                String msg = t.getMessage();
                throw new LlamaCppJnaException(
                        "Library '" + libraryName + "' is not available"
                                + (msg == null || msg.trim().isEmpty() ? "" : ", error: " + msg),
                        t);
            }
        }
    }

    /** @return true if the instance has been successfully loaded. */
    @Override
    public boolean isAvailable() {
        return instance != null;
    }

    /** @return the last Throwable encountered during loading, or null if none. */
    @Override
    public Throwable getLoadThrowable() {
        return loadThrowable;
    }

    /** @return the last error message, or null if none. */
    @Override
    public String getLoadErrorMessage() {
        return loadThrowable == null ? null : loadThrowable.getMessage();
    }

    @Override
    public String getLibraryName() {
        return libraryName;
    }

    @Override
    public Map<String, Object> getLibraryOptions() {
        return libraryOptions; // safe to return, it's unmodifiable
    }

    @Override
    public Class<T> getLibraryClass() {
        return libraryClass;
    }
}

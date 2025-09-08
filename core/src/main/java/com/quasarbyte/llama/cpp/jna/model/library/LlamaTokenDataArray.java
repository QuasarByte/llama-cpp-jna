package com.quasarbyte.llama.cpp.jna.model.library;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Plain Java representation of {@code llama_token_data_array}.
 */
public class LlamaTokenDataArray {

    private final List<LlamaTokenData> entries;
    private long selectedIndex = -1;
    private boolean sorted;

    public LlamaTokenDataArray() {
        this.entries = new ArrayList<>();
    }

    public LlamaTokenDataArray(List<LlamaTokenData> entries) {
        this.entries = new ArrayList<>(entries != null ? entries : Collections.emptyList());
    }

    public List<LlamaTokenData> getEntries() {
        return Collections.unmodifiableList(entries);
    }

    public LlamaTokenDataArray addEntry(LlamaTokenData data) {
        entries.add(data);
        return this;
    }

    public long getSelectedIndex() {
        return selectedIndex;
    }

    public LlamaTokenDataArray setSelectedIndex(long selectedIndex) {
        this.selectedIndex = selectedIndex;
        return this;
    }

    public boolean isSorted() {
        return sorted;
    }

    public LlamaTokenDataArray setSorted(boolean sorted) {
        this.sorted = sorted;
        return this;
    }

    /**
     * Creates a native structure to be passed to llama.cpp functions.
     */
    public LlamaTokenDataArrayNative toNative() {
        LlamaTokenDataArrayNative nativeArray = new LlamaTokenDataArrayNative();

        LlamaTokenDataNative[] nativeEntries = createNativeEntries(entries);
        nativeArray.setEntries(nativeEntries);
        nativeArray.selected = selectedIndex;
        nativeArray.setSorted(sorted);
        nativeArray.write();

        return nativeArray;
    }

    private static LlamaTokenDataNative[] createNativeEntries(List<LlamaTokenData> source) {
        if (source.isEmpty()) {
            return new LlamaTokenDataNative[0];
        }

        LlamaTokenData firstEntry = source.get(0);
        LlamaTokenDataNative first = new LlamaTokenDataNative();
        first.id = firstEntry.getId();
        first.logit = firstEntry.getLogit();
        first.p = firstEntry.getProbability();

        LlamaTokenDataNative[] nativeEntries = (LlamaTokenDataNative[]) first.toArray(source.size());

        for (int i = 0; i < source.size(); i++) {
            LlamaTokenData entry = source.get(i);
            nativeEntries[i].id = entry.getId();
            nativeEntries[i].logit = entry.getLogit();
            nativeEntries[i].p = entry.getProbability();
            nativeEntries[i].write();
        }

        return nativeEntries;
    }

    /**
     * Wraps the provided native view for easier debugging.
     */
    public static LlamaTokenDataArray fromNative(LlamaTokenDataArrayNative nativeArray) {
        LlamaTokenDataArray result = new LlamaTokenDataArray();
        if (nativeArray == null) {
            return result;
        }

        for (LlamaTokenDataNative entry : nativeArray.getEntries()) {
            result.addEntry(LlamaTokenData.fromNative(entry));
        }
        result.setSelectedIndex(nativeArray.selected);
        result.setSorted(nativeArray.isSorted());
        return result;
    }
}


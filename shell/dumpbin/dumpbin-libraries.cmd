SET PATH=%PATH;"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"

call dumpbin /exports ggml.dll > ggml-dll-dumpbin.txt
call dumpbin /exports ggml-base.dll > ggml-base-dll-dumpbin.txt
call dumpbin /exports ggml-cpu-alderlake.dll > ggml-cpu-alderlake-dll-dumpbin.txt
call dumpbin /exports ggml-cpu-haswell.dll > ggml-cpu-haswell-dll-dumpbin.txt
call dumpbin /exports ggml-cpu-icelake.dll > ggml-cpu-icelake-dll-dumpbin.txt
call dumpbin /exports ggml-cpu-sandybridge.dll > ggml-cpu-sandybridge-dll-dumpbin.txt
call dumpbin /exports ggml-cpu-sapphirerapids.dll > ggml-cpu-sapphirerapids-dll-dumpbin.txt
call dumpbin /exports ggml-cpu-skylakex.dll > ggml-cpu-skylakex-dll-dumpbin.txt
call dumpbin /exports ggml-cpu-sse42.dll > ggml-cpu-sse42-dll-dumpbin.txt
call dumpbin /exports ggml-cpu-x64.dll > ggml-cpu-x64-dll-dumpbin.txt
call dumpbin /exports ggml-rpc.dll > ggml-rpc-dll-dumpbin.txt
call dumpbin /exports libcurl.dll > libcurl-dll-dumpbin.txt
call dumpbin /exports libcurl-x64.dll > libcurl-x64-dll-dumpbin.txt
call dumpbin /exports libomp140.x86_64.dll > libomp140.x86_64-dll-dumpbin.txt
call dumpbin /exports llama.dll > llama-dll-dumpbin.txt
call dumpbin /exports zlib1.dll > zlib1-dll-dumpbin.txt

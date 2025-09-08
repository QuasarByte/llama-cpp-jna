SET JAVA_HOME=C:\opt\java\jdk-25

SET PATH=%JAVA_HOME%\bin;%PATH%
SET PATH=%PATH%;C:\opt\llama.cpp-b6527-bin
SET GGML_BACKEND_PATH=C:\opt\llama.cpp-b6527-bin

cd /d C:\projects\llama-cpp-jna

call mvn compile -pl examples

call mvn -pl examples exec:java ^
  -Dexec.mainClass=com.quasarbyte.llama.cpp.jna.examples.simplechat.SimpleChat ^
  -Dexec.args="-m C:\opt\models\Qwen3-8B-Q8_0.gguf -c 32768 -ngl 100" ^
  -Dexec.includeProjectDependencies=true
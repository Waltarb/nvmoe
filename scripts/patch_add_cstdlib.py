path = '/root/projects/llama.cpp/src/llama-model-loader.cpp'
with open(path) as f:
    content = f.read()

old = '#include "llama-model-loader.h"\n'
assert old in content
content = content.replace(old, old + '#include <cstdlib>\n', 1)
with open(path, 'w') as f:
    f.write(content)
print('cstdlib include added')

# save as: extract_imports.py
import os
import ast
from collections import defaultdict

project_dir = "/Users/gaozhen/d_pan/Documents/source_code/UNI"  # 修改为你的项目目录路径

imports = set()

for root, _, files in os.walk(project_dir):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read(), filename=filepath)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for n in node.names:
                                imports.add(n.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imports.add(node.module.split('.')[0])
                except Exception as e:
                    print(f"⚠️ Failed to parse {filepath}: {e}")

print("# Estimated dependencies:")
for name in sorted(imports):
    print(name)

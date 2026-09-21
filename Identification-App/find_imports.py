"""
Run this on your machine (inside the venv, from the project root) as:
    python find_imports.py

Walks engine/ and prints every top-level third-party package it imports,
so you can check the build script's --include-package list covers all of
them in one go instead of discovering them one crash at a time.
"""
import ast
import pathlib
import sys

STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()

engine_dir = pathlib.Path("engine")
local_modules = {p.stem for p in engine_dir.rglob("*.py")}
local_packages = {p.name for p in engine_dir.iterdir() if p.is_dir()}

found = set()
for pyfile in engine_dir.rglob("*.py"):
    try:
        tree = ast.parse(pyfile.read_text(encoding="utf-8"), filename=str(pyfile))
    except SyntaxError as e:
        print(f"[skip, syntax error] {pyfile}: {e}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])

third_party = sorted(found - STDLIB - local_modules - local_packages)
print("Top-level third-party packages imported somewhere under engine/:\n")
for name in third_party:
    print(f"  {name}")
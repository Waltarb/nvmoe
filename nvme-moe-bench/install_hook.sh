#!/usr/bin/env bash
# install_hook.sh — Install nvmoe (3-Tier NVMe Cache hook) and FreeToken patches
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== nvmoe: FreeToken 3-Tier NVMe Integration Setup ==="

# 1. Determine target Python / venv
if [ -n "$VIRTUAL_ENV" ]; then
    PY_BIN="$VIRTUAL_ENV/bin/python3"
elif [ -d "$HOME/.freetoken/venv" ]; then
    PY_BIN="$HOME/.freetoken/venv/bin/python3"
else
    PY_BIN="$(command -v python3)"
fi

if [ ! -x "$PY_BIN" ]; then
    echo "[-] Error: Could not find a valid python3 interpreter."
    echo "    Please activate your FreeToken virtual environment or set VIRTUAL_ENV."
    exit 1
fi

echo "[+] Using Python: $PY_BIN"

# 2. Locate site-packages and freetoken
SITE_PACKAGES="$("$PY_BIN" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || true)"
if [ -z "$SITE_PACKAGES" ] || [ ! -d "$SITE_PACKAGES" ]; then
    SITE_PACKAGES="$("$PY_BIN" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
fi

echo "[+] Target site-packages: $SITE_PACKAGES"

FREETOKEN_DIR="$SITE_PACKAGES/freetoken"
if [ ! -d "$FREETOKEN_DIR" ]; then
    echo "[-] Warning: freetoken package not found in $SITE_PACKAGES."
    echo "    Make sure freetoken is installed (e.g. pip install freetoken-...whl)."
else
    echo "[+] Found freetoken installation: $FREETOKEN_DIR"

    # 3. Apply patches if needed
    PATCH_FILE="$SCRIPT_DIR/patches/freetoken-0.1.2.patch"
    if [ -f "$PATCH_FILE" ]; then
        if grep -q "qwen4_ngram" "$FREETOKEN_DIR/models/qwen4_exp/ple_disk.py" 2>/dev/null; then
            echo "[+] FreeToken Qwen3.8 / PLE disk patches already applied."
        else
            echo "[*] Applying patches/freetoken-0.1.2.patch to $FREETOKEN_DIR..."
            if command -v patch >/dev/null 2>&1; then
                (cd "$SITE_PACKAGES" && patch -p1 -N -s < "$PATCH_FILE") && \
                    echo "[+] Successfully applied FreeToken patches." || \
                    echo "[!] Note: Patch application returned non-zero (may be partially applied or modified)."
            else
                echo "[-] Warning: 'patch' command not found. Please install patch (e.g. apt install patch) to apply FreeToken vendor fixes."
            fi
        fi
    fi
fi

# 4. Install sitecustomize.py hook
TARGET_HOOK="$SITE_PACKAGES/sitecustomize.py"
SOURCE_HOOK="$SCRIPT_DIR/sitecustomize.py"

if [ -f "$TARGET_HOOK" ] && grep -q "FREETOKEN_NVME_TIER" "$TARGET_HOOK"; then
    echo "[+] sitecustomize.py hook is already installed in $SITE_PACKAGES."
else
    if [ -f "$TARGET_HOOK" ]; then
        echo "[*] Appending NVMe tier hook to existing $TARGET_HOOK..."
        echo "" >> "$TARGET_HOOK"
        cat "$SOURCE_HOOK" >> "$TARGET_HOOK"
    else
        echo "[*] Installing sitecustomize.py hook into $SITE_PACKAGES..."
        cp "$SOURCE_HOOK" "$TARGET_HOOK"
    fi
    echo "[+] Hook installed successfully."
fi

echo ""
echo "=== Integration Setup Complete ==="
echo "To launch FreeToken with the 3-Tier NVMe MoE cache, run:"
echo "  export FREETOKEN_NVME_TIER=1"
echo "  ./ft_serve_qwen38_background.sh"

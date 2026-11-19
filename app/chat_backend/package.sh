#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qonnect-chat-backend.XXXXXX")"
STAGING_DIR="${SCRIPT_DIR}/.build-staging-$$"
BACKUP_DIR="${SCRIPT_DIR}/.build-backup-$$"
BUILD_DIR="${SCRIPT_DIR}/build"

cleanup() {
    if [[ -d "$BACKUP_DIR" && ! -d "$BUILD_DIR" ]]; then
        mv "$BACKUP_DIR" "$BUILD_DIR"
    fi
    rm -rf "$TEMP_DIR" "$STAGING_DIR"
}
trap cleanup EXIT

command -v uv >/dev/null 2>&1 || {
    echo "ERROR: uv no está instalado."
    exit 1
}

echo "=== Exportando dependencias bloqueadas ==="
uv export \
    --locked \
    --no-dev \
    --no-emit-project \
    --format requirements.txt \
    --directory "$SCRIPT_DIR" \
    > "$TEMP_DIR/requirements.txt"

echo "=== Instalando wheels Python 3.13 / ARM64 ==="
uv pip install \
    --target "$STAGING_DIR" \
    --python-version 3.13 \
    --python-platform aarch64-manylinux_2_17 \
    --only-binary=:all: \
    -r "$TEMP_DIR/requirements.txt"

echo "=== Copiando aplicación ==="
cp "$SCRIPT_DIR/handler.py" "$STAGING_DIR/handler.py"
cp "$SCRIPT_DIR/run.sh" "$STAGING_DIR/run.sh"
chmod +x "$STAGING_DIR/run.sh"

echo "=== Publicando paquete ==="
if [[ -d "$BUILD_DIR" ]]; then
    mv "$BUILD_DIR" "$BACKUP_DIR"
fi

if ! mv "$STAGING_DIR" "$BUILD_DIR"; then
    if [[ -d "$BACKUP_DIR" ]]; then
        mv "$BACKUP_DIR" "$BUILD_DIR"
    fi
    echo "ERROR: no se pudo publicar el paquete."
    exit 1
fi

rm -rf "$BACKUP_DIR"

echo "=== Empaquetado completado ==="
du -sh "$BUILD_DIR"
echo "Archivos principales:"
ls -lh "$BUILD_DIR/handler.py" "$BUILD_DIR/run.sh"

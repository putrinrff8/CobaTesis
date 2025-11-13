#!/usr/bin/env bash
set -euo pipefail

# Script setup otomatis untuk services/backend dan services/frontend
# Folder services/ml-pipeline diabaikan

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICES_DIR="$ROOT_DIR/services"
BACKEND_DIR="$SERVICES_DIR/backend"
FRONTEND_DIR="$SERVICES_DIR/frontend"
ML_DIR="$SERVICES_DIR/ml-pipeline"

DRY_RUN=0

print_usage() {
  cat <<USAGE
Usage: $(basename "$0") [--dry-run|-n] [--help|-h]

Opsi:
  --dry-run, -n    Tampilkan langkah tanpa mengeksekusi perintah
  --help, -h       Tampilkan pesan ini
Default:
  Menjalankan setup otomatis untuk backend dan frontend.
USAGE
}

log() {
  echo -e "\n==> $*"
}

run_cmd() {
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "(dry-run) $*"
  else
    echo "+ $*"
    eval "$@"
  fi
}

# Parse argumen
if [ "${1-}" = "--help" ] || [ "${1-}" = "-h" ]; then
  print_usage
  exit 0
fi

if [ "${1-}" = "--dry-run" ] || [ "${1-}" = "-n" ]; then
  DRY_RUN=1
fi

# Buat README di root proyek (menimpa jika sudah ada)
log "Menulis README.md di root proyek"
cat > "$ROOT_DIR/README.md" <<'README'
# Setup awal layanan (backend & frontend)

Deskripsi singkat
-----------------
Panduan ini menjelaskan langkah-langkah setup awal untuk layanan `backend` dan `frontend`
yang berada di direktori `services`. Folder pipeline ML (`services/ml-pipeline`) diabaikan
oleh script ini.

Struktur proyek (relevan)
------------------------
- `services/backend`  - kode backend (Python)
- `services/frontend` - kode frontend (Node.js / Bun / yarn)
- `services/ml-pipeline` - pipeline ML (tidak termasuk pada setup ini)

Prasyarat
---------
- Sistem operasi: Unix-like (Linux, macOS) atau Windows + WSL
- Python 3.8+ tersedia sebagai `python3` atau `python`
- Node.js + npm tersedia bila menggunakan npm
- Bun tersedia bila menggunakan Bun (opsional)
- Git tersedia bila diperlukan

Cara penggunaan
---------------
1. Jadikan file ini eksekusi:
   ```
   chmod +x setup.sh
   ```
2. Jalankan setup otomatis:
   ```
   ./setup.sh
   ```
3. Menjalankan mode dry-run (tampilkan langkah tanpa mengeksekusi):
   ```
   ./setup.sh --dry-run
   ```

Ringkasan tindakan script
-------------------------
- Menyiapkan virtual environment di `services/backend/.venv` bila belum ada.
- Menginstal dependencies Python dari `requirements.txt` bila file tersebut ada.
- Memilih package manager frontend berdasarkan keberadaan file lock:
  - `bun.lock`  -> `bun install`
  - `yarn.lock` -> `yarn install`
  - `package-lock.json` -> `npm ci`
  - fallback -> `npm install`
- Menjalankan `npm run build`/`yarn build`/`bun build` bila script `build` tersedia di `package.json`.
- Tidak melakukan tindakan apapun pada `services/ml-pipeline`.

Troubleshooting singkat
-----------------------
- Jika Python tidak terdeteksi, periksa instalasi Python atau gunakan manajer versi (pyenv).
- Jika permission error muncul, beri izin eksekusi pada `setup.sh` dan pastikan akses filesystem.
- Jika dependency frontend tidak terinstal, cek versi Node.js dan hak akses pada `node_modules`.

Kontak pengembang
-----------------
Informasi detail tentang tiap layanan dapat ditemukan pada folder masing-masing:
- `services/backend`
- `services/frontend`

README dibuat otomatis oleh `setup.sh`.
README

# START SETUP
log "Root proyek: $ROOT_DIR"
log "Direktori services: $SERVICES_DIR"

# Backend setup
if [ -d "$BACKEND_DIR" ]; then
  log "Menyiapkan backend di: $BACKEND_DIR"
  (
    cd "$BACKEND_DIR"
    # Tentukan perintah python yang tersedia
    PY_CMD=""
    if command -v python3 >/dev/null 2>&1; then
      PY_CMD="python3"
    elif command -v python >/dev/null 2>&1; then
      PY_CMD="python"
    fi

    if [ -z "$PY_CMD" ]; then
      log "Python tidak ditemukan di PATH; melewati instalasi backend"
    else
      # Buat virtualenv bila belum ada
      if [ ! -d ".venv" ]; then
        log "Membuat virtual environment (.venv)"
        run_cmd "$PY_CMD -m venv .venv"
      else
        log "Virtual environment .venv sudah ada"
      fi

      ACTIVATE=".venv/bin/activate"
      if [ -f "$ACTIVATE" ]; then
        # Gunakan venv untuk instalasi paket
        log "Menggunakan virtualenv untuk instalasi paket"
        # pip path
        PIP_CMD=".venv/bin/pip"
        if [ -f "requirements.txt" ]; then
          log "Menemukan requirements.txt, menginstal dependency Python"
          run_cmd "$PIP_CMD install --upgrade pip"
          run_cmd "$PIP_CMD install -r requirements.txt"
        else
          # Cek alternatif: pyproject.toml
          if [ -f "pyproject.toml" ]; then
            log "Menemukan pyproject.toml"
            # Gunakan pip install . sebagai fallback
            if [ -f "setup.cfg" ] || [ -f "setup.py" ]; then
              log "Menemukan setup.py/setup.cfg, menjalankan pip install ."
              run_cmd "$PIP_CMD install -e ."
            else
              log "Tidak ditemukan requirements.txt atau setup.py. Lewati instalasi Python."
            fi
          else
            log "Tidak ditemukan requirements.txt atau pyproject.toml. Lewati instalasi Python."
          fi
        fi
      else
        log "Virtualenv tidak valid, melewati instalasi backend"
      fi
    fi
  )
else
  log "Direktori backend tidak ditemukan; melewati langkah backend"
fi

# Frontend setup
if [ -d "$FRONTEND_DIR" ]; then
  log "Menyiapkan frontend di: $FRONTEND_DIR"
  (
    cd "$FRONTEND_DIR"

    # Pilih package manager
    PM=""
    if [ -f "bun.lock" ] && command -v bun >/dev/null 2>&1; then
      PM="bun"
    elif [ -f "yarn.lock" ] && command -v yarn >/dev/null 2>&1; then
      PM="yarn"
    elif [ -f "package-lock.json" ] && command -v npm >/dev/null 2>&1; then
      PM="npm-ci"
    elif command -v npm >/dev/null 2>&1; then
      PM="npm"
    fi

    if [ -z "$PM" ]; then
      log "Tidak ditemukan package manager yang tersedia (bun/yarn/npm). Lewati instalasi frontend"
    else
      case "$PM" in
        bun)
          log "Menggunakan Bun untuk instalasi"
          run_cmd "bun install"
          ;;
        yarn)
          log "Menggunakan Yarn untuk instalasi"
          run_cmd "yarn install --frozen-lockfile"
          ;;
        npm-ci)
          log "Menggunakan npm ci untuk instalasi"
          run_cmd "npm ci"
          ;;
        npm)
          log "Menggunakan npm install sebagai fallback"
          run_cmd "npm install"
          ;;
      esac

      # Jalankan build bila script build tersedia di package.json
      if [ -f "package.json" ]; then
        if grep -q '"build"' package.json; then
          log "Menemukan script 'build' di package.json, menjalankan build"
          if [ "$PM" = "bun" ]; then
            run_cmd "bun run build"
          elif [ "$PM" = "yarn" ]; then
            run_cmd "yarn build"
          else
            run_cmd "npm run build --silent"
          fi
        else
          log "Tidak ditemukan script 'build' di package.json; melewati langkah build"
        fi
      fi
    fi
  )
else
  log "Direktori frontend tidak ditemukan; melewati langkah frontend"
fi

log "Langkah setup selesai. Folder ML pipeline ($ML_DIR) sengaja diabaikan dalam proses ini."

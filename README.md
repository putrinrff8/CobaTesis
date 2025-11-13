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

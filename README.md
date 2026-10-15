<p align="center"><img src="docs/logo.png" width="140" alt="Photocure logo"></p>

# Photocure

Intelligent, offline photo library curation engine with hardware acceleration for Apple Silicon (Mac GPU / Metal), NVIDIA (CUDA), AMD (ROCm), Intel, and CPU. It scores image quality, detects faces, classifies photos with OpenCLIP, groups bursts, and organizes everything into a clean folder structure with 100% reversible undo.

---

## Setup (First time)

### Option 1: Debian / Ubuntu (.deb package)

Download and install the latest `.deb` directly:

```bash
# Download the latest .deb package
curl -LO https://github.com/safiyu/photocure/releases/latest/download/photocure_amd64.deb

# Install with apt
sudo apt install ./photocure_amd64.deb

# Run setup to detect GPU and prepare models
photocure setup
```

*(You can also download versioned `.deb` files directly from [GitHub Releases](https://github.com/safiyu/photocure/releases)).*

### Option 2: From Source

```bash
cd photocure
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
photocure setup
```

`photocure setup` detects your hardware and configures the optimal acceleration:
- **Apple Silicon (M1/M2/M3/M4):** Native Apple GPU acceleration via Metal Performance Shaders (MPS).
- **NVIDIA GPU:** CUDA acceleration.
- **AMD & Intel GPUs:** ROCm and XPU acceleration.
- **CPU:** Multi-threaded CPU processing.

| Option | What it does |
|---|---|
| `--dry-run` | Show what would be installed without downloading anything |
| `--yes`, `-y` | Do not ask confirmation before downloading |
| `--no-faces` | Skip face detection (InsightFace) |
| `--cpu-only` | Ignore any GPU and force CPU build |
| `--reinstall` | Force reinstalling PyTorch (e.g. after changing GPU) |
| `--no-open` | Start server without opening the browser |

---

## Start (Daily use)

To open Photocure anytime after the initial setup without reinstalling or checking dependencies:

```bash
photocure start
# or:
./start.sh
```

Both start the dashboard and automatically open `http://localhost:8080/` in your default browser.

---

## Serve

Launch the web review dashboard directly:

```bash
photocure serve [OPTIONS]
```

### Options

| Option | Default | Description |
|---|---|---|
| `--port`, `-p` | `8080` | Port to bind (auto-picks next available port if busy) |
| `--host` | `127.0.0.1` | Host address to bind |
| `--db` | `photocure.db` | Path to the SQLite database |
| `--open` / `--no-open` | `--open` | Open the browser automatically when ready |
| `--reload` | `False` | Auto-reload server on code changes |

### Web dashboard features

- **Import:** Step-by-step import wizard (scan, analyze, detect bursts) with built-in folder picker.
- **Review:** Filter by category (People, Animals, Food, Scenery, Architecture, etc.), bursts, highlights, and junk.
- **Actions:** Safe dry-run plan, execution, and 1-click byte-for-byte rollback.

---

## Docker

Run Photocure in a self-contained container using the pre-built image from Docker Hub (`safiyu/photocure:latest`):

```bash
PHOTOS_DIR=/path/to/photos OUTPUT_DIR=/path/to/organized docker compose up
```

Or using `docker run`:

```bash
docker run -d \
  --name photocure \
  -p 127.0.0.1:8080:8080 \
  -v /path/to/photos:/photos:rw \
  -v /path/to/organized:/output:rw \
  -v curator-data:/data \
  safiyu/photocure:latest
```

Then open `http://localhost:8080/` in your browser.

- On first run, the container detects your hardware and installs the matching AI libraries into the persistent `curator-data` volume. Later starts reuse this volume and launch immediately.
- **Volumes:**
  - `PHOTOS_DIR`: Path to your photo library (mounts at `/photos`).
  - `OUTPUT_DIR`: Path where curated photos will be saved (mounts at `/output`).
  - `curator-data`: Docker volume storing database, thumbnails, and model weights.

### GPU Acceleration in Docker

Add the compose file for your GPU:

- **NVIDIA GPU:**
  ```bash
  PHOTOS_DIR=/path/to/photos OUTPUT_DIR=/path/to/organized docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
  ```
  *(Requires NVIDIA Container Toolkit)*

- **AMD GPU:**
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.rocm.yml up
  ```

- **Intel GPU:**
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.intel.yml up
  ```

- **Apple Silicon (Mac):** Docker Desktop on macOS runs Linux in a VM without Metal GPU pass-through. To use your Mac GPU, run Photocure natively with `photocure start`.

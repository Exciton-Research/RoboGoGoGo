# Install on macOS or Linux

The default install runs JAX on the CPU and serves a built Studio from one local
Python process. It needs no robot, ROS installation, CUDA toolkit, or GPU.

## Supported setup

- **macOS:** Apple Silicon with a native ARM64 Python interpreter. Avoid running
  the terminal or Python under Rosetta for this setup.
- **Linux:** x86_64 or ARM64 with a compatible manylinux/glibc environment. Ubuntu
  24.04 is a straightforward starting point; Alpine/musl is outside this path.
- **Python:** 3.11 or newer; 3.12 is the CI baseline. Pip resolves dependencies for
  your interpreter. Very new Python releases can precede binary wheel availability.
- **Node.js:** 22.6 or newer, using a supported LTS release, with npm. Node is
  needed to build Studio, not to serve it after building.
- **Browser:** a current desktop browser with WebGL support for simulation and
  camera capture. Headless numerical runs do not need a browser.

Current upstream JAX CPU wheels cover Apple Silicon macOS and Linux x86_64/ARM64;
see the [JAX installation matrix](https://docs.jax.dev/en/latest/installation.html).
Intel macOS is not supported by this setup script. Use a supported Linux machine
or investigate an upstream source build separately; an old JAX pin is not a
verified compatibility path for this repo. Use CPU JAX on macOS; GPU setup is
outside this guide. Select a supported release from the
[Node.js release list](https://nodejs.org/en/about/previous-releases).

## 1. Install prerequisites

### macOS (Apple Silicon)

If you already use [Homebrew](https://brew.sh/):

```sh
brew install python@3.12 node@22 git
export PATH="$(brew --prefix node@22)/bin:$(brew --prefix python@3.12)/libexec/bin:$PATH"
python3.12 --version
node --version
npm --version
```

Add the PATH setting to your shell configuration if you want it to persist.
Alternatively, install Python and Node from their official installers. The
repository setup does not install Homebrew or change your shell files.

### Ubuntu / Debian Linux

```sh
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip
python3 --version
```

Your Python must be at least 3.11. Ubuntu 24.04 supplies Python 3.12. Install a
supported Node LTS using the [official Node.js download instructions](https://nodejs.org/en/download),
then verify:

```sh
node --version
npm --version
```

Distribution packages may provide an older Node release; check before setup.
On other Linux distributions install equivalent Git, Python, and venv packages.

## 2. Clone and set up

```sh
git clone https://github.com/Exciton-Research/robogogogo.git
cd robogogogo
bash scripts/setup.sh
```

On macOS, select the Homebrew Python explicitly if needed:

```sh
PYTHON=python3.12 bash scripts/setup.sh
```

The script checks the platform and tools, creates `.venv`, installs the `studio`
and `perception` extras, installs locked npm dependencies, builds/packages the
static UI, and runs a one-step simulation smoke check. It stops on errors rather
than reporting a partial installation as complete. It can be run from any working
directory using the script's full path.

For development tools use `bash scripts/setup.sh --dev`. To choose a virtualenv
outside the checkout, set `ROBOGOGOGO_VENV=/absolute/path/to/venv`. Subsequent
commands must use that environment's executables instead of `.venv/bin/…`.

## 3. Start Studio

From the repo root:

```sh
.venv/bin/robogogogo-studio
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Wait for `Application startup
complete` in the terminal; the first JAX compilation is slower than later steps.
The server stays in this terminal. Ctrl+C shuts it down.

To keep recordings in a fixed location or use a different port:

```sh
.venv/bin/robogogogo-studio --port 8876 --projects "$HOME/robogogogo-runs"
```

Then open [http://127.0.0.1:8876](http://127.0.0.1:8876). The built UI uses that
same host and port. The server binds to loopback; this is one shared local
workspace, not a multi-user hosted service.

## Manual installation

These are the steps performed by the full setup script, without its tool checks
and simulation smoke test:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[studio,perception]'
npm ci --prefix studio
npm run build --prefix studio
.venv/bin/python -m robogogogo.server
```

Quote the Python extras so zsh does not expand the brackets. `perception` adds
OpenCV for stereo analysis; `dev` adds pytest, Ruff, and HTTP client test support.
Use `.[studio,perception,dev]` for all development checks.

## Headless numerical use

Install just the engine; Python is sufficient:

```sh
bash scripts/setup.sh --headless
.venv/bin/python -m robogogogo --headless --steps 100 --export /tmp/mobile-run.npz
```

The JSON result reports elapsed simulation time and mobile diagnostics. The NPZ
contains numerical history, model parameters, and robot identity, without images.
Add `--dev` to the setup command if you also want Python test tools. Headless
steps do not start the HTTP server or produce camera RGB.

## Updating and removing

After updating a clean checkout with `git pull`, rerun `bash scripts/setup.sh`
(include `--dev` if you use test tools). This reinstalls Python requirements and
rebuilds Studio with the checked-in npm lockfile. A Python editable install picks
up Python edits, but frontend edits still need a rebuild and server restart.

Recordings survive setup. Back up `runs/studio/` or your `--projects` directory
before changing robot revisions. Historical archives remain inspectable; current
learning validation rejects a mismatched robot revision.

To uninstall, stop the server and remove this checkout's `.venv`, `studio/node_modules`,
`studio/.next`, `studio/out`, and `src/robogogogo/web`. Keep recordings if you need
them. No system package removal is required by the repository setup.

## Troubleshooting

### `python3` is too old or `venv` is missing

Choose an installed supported interpreter with `PYTHON=python3.12`. On Ubuntu,
install `python3-venv` (or the venv package matching your chosen Python). If `.venv`
was created by an old interpreter, create a new environment at a different path
using `ROBOGOGOGO_VENV`; an existing venv is reused, not silently replaced.

### Pip reports `externally-managed-environment`

Use `.venv/bin/python -m pip`, not system pip. The setup script uses a virtualenv;
do not work around this by installing into system Python with sudo.

### `jaxlib` has no matching distribution

Check `python3 -c 'import platform; print(platform.system(), platform.machine())'`
and your Python version against the upstream wheel matrix. On Apple Silicon,
an `x86_64` result usually means an Intel interpreter or Rosetta. On Linux, use
a supported architecture and glibc distribution. Try the CI Python baseline
3.12 in a fresh environment if your interpreter is newer than available wheels.

### npm rejects the lockfile or Node version

Check `node --version` is at least 22.6, and run `npm ci --prefix studio` from the
repo root. Use the committed lockfile. If install fails, read the first error;
the build cannot run until dependency installation succeeds.

### Browser shows “Studio assets are not built”

Run `npm run build --prefix studio`, then restart Python. The build copies the
static export into `src/robogogogo/web/`. Running `next start` is not the packaged
production path. Startup decides whether static assets are present.

### Port 8765 is already in use

Start with `--port 8876` and use the matching URL. Do not stop an unrelated process
just to free the default port.

### Simulation is offline or the view is blank

Check the Python terminal for startup errors, and wait for compilation to finish.
The `/simulation/` view uses `/api/ws` on the same origin in a production build.
Try a WebGL-capable browser and check its console if the canvas cannot initialize.
On a remote Linux host, forward the loopback port with SSH and render in your
local browser; no display server is needed for the Python engine.

### A movement is rejected or the episode has ended

Guards can reject colliding, unstable, or moving-base arm goals. Pause, brake and
let the robot settle before planning; reset a failed simulation before continuing.
Robot debug controls invalidate visual episodes. Choose **Start new episode**
before recording again. See the [walkthrough](getting-started.md).

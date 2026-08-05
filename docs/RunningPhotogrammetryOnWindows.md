# Running Photogrammetry on Windows

The Photogrammetry extension was built for Linux, where the reconstruction engine runs
in a Docker container. On Windows there is no Docker in this workflow. Instead you
install two OpenDroneMap programs natively, once, and point Slicer at them.

Everything here is a manual, one-time setup. The extension never downloads or installs
these programs for you.

> **Status: verified through launching NodeODM.** Confirmed on Windows 11 with Slicer
> 5.12.3, ODM 3.6.1 and NodeODM v2.2.3: Steps 1-5 all work, both path fields prefill
> themselves, and **Launch NodeODM** reports the node up and serving. Getting there
> required [Step 4b](#step-4b---patch-one-file-in-nodeodm-required), which was found by
> testing rather than by reading the sources. A full reconstruction and everything from
> "Running a reconstruction" onward is still untested. If a step below is wrong, please
> open an issue.

## Table of Contents

1. [What you need](#what-you-need)
   - [Check the VC++ runtime first](#check-the-vc-runtime-first)
2. [Step 1 - Install 3D Slicer and the PyTorch extension](#step-1---install-3d-slicer-and-the-pytorch-extension)
3. [Step 2 - Get the Photogrammetry extension](#step-2---get-the-photogrammetry-extension)
4. [Step 3 - Install ODM](#step-3---install-odm)
5. [Step 4 - Install NodeODM](#step-4---install-nodeodm)
6. [Step 5 - Point Slicer at both](#step-5---point-slicer-at-both)
7. [Running a reconstruction](#running-a-reconstruction)
8. [PhotoMasking and VideoMasking](#photomasking-and-videomasking)
9. [Troubleshooting](#troubleshooting)
10. [What is different from Linux](#what-is-different-from-linux)

---

## What you need

- 64-bit Windows 10 or 11.
- The **Microsoft Visual C++ 2015-2022 Redistributable (x64), version 14.40 or newer.**
  This is a hard requirement, and the one most likely to bite you - see below.
- About **12 GB** of free disk space for the software, plus room for your projects.
  Reconstructions are large; budget several GB per dataset.
- **16 GB RAM minimum**, more is better. Photogrammetry is memory-hungry.
- An **NVIDIA GPU is optional but strongly recommended.** Only the driver is needed -
  no CUDA toolkit, and no NVIDIA Container Toolkit. If you have no NVIDIA card,
  everything still works on the CPU, just slower.
- A network connection for the downloads below.

You do **not** need Docker, WSL, git, or admin rights on the Slicer side.

### Check the VC++ runtime first

ODM 3.6.1 is built with the Visual Studio 2022 toolset. Against an older runtime its
GDAL binaries do not fail gracefully - they crash on load, and every reconstruction dies
instantly with a bare `Processing failed (3221225477)` and an empty log. A machine that
has only ever had older Visual C++ redistributables installed (14.22, from 2019, is a
common one to be stuck on) will look fine until the very first task fails.

`ODM_Setup` installs the right runtime itself, but that step needs elevation - so if you
decline the UAC prompt, you get exactly this. Checking takes a second:

```powershell
(Get-Item C:\Windows\System32\MSVCP140.dll).VersionInfo.FileVersion
```

If that reports below **14.40**, or the file is missing, install
[VC_redist.x64.exe 14.44.35211](https://download.visualstudio.microsoft.com/download/pr/9b0d1fa5-c16d-4ee8-97f0-c2734086ece8/CC0FF0EB1DC3F5188AE6300FAEF32BF5BEEBA4BDD6E8E445A9184072096B713B/VC_redist.x64.exe)
(24 MB) and reboot before you start. Doing it now costs a minute; discovering it later
costs a full masking run and a failed task.

That is the exact build this document was verified against. Microsoft's evergreen link,
<https://aka.ms/vs/17/release/vc_redist.x64.exe>, always serves the current
release - newer is fine here, since the runtime is backward compatible.

---

## Step 1 - Install 3D Slicer and the PyTorch extension

1. Install 3D Slicer from [download.slicer.org](https://download.slicer.org).
2. Open Slicer, go to the **Extensions Manager**, and install:
   - **PyTorch**
   - **SlicerMorph**

Restart Slicer when asked.

---

## Step 2 - Get the Photogrammetry extension

Photogrammetry is not in the Windows extension catalogue, so it is loaded from a local
copy of the source rather than installed through the Extensions Manager.

1. Download the `windows-only` branch as a zip:
   <https://github.com/SlicerMorph/SlicerPhotogrammetry/archive/refs/heads/windows-only.zip>
2. Extract it somewhere permanent, for example `C:\SlicerPhotogrammetry`.
   Slicer reads the modules from this folder every time it starts, so do not put it in
   Downloads or a temp folder.
3. In Slicer, open **Edit > Application Settings > Modules**.
4. Drag the following four folders into the **Additional module paths** list (or use the
   `>>` button to add each one):

   ```
   C:\SlicerPhotogrammetry\PhotoMasking
   C:\SlicerPhotogrammetry\VideoMasking
   C:\SlicerPhotogrammetry\ClusterPhotos
   C:\SlicerPhotogrammetry\ODM
   ```

5. Restart Slicer. The modules appear under **SlicerMorph > Photogrammetry** in the
   module list.

---

## Step 3 - Install ODM

ODM is the reconstruction engine. OpenDroneMap publishes a free native Windows
installer on GitHub.

1. Download the installer this document was verified with, **ODM 3.6.1**:

   <https://github.com/OpenDroneMap/ODM/releases/download/v3.6.1/ODM_Setup_3.6.1.exe>

   234 MB, published 28 July 2026. That link is pinned to the release tag and will keep
   serving this exact file.
2. Nothing here assumes 3.6.1 in particular, so a newer installer is worth trying - but
   note that **not every ODM release ships a Windows installer.** `v3.6.0` and `v3.5.6`,
   for instance, have source archives only. Check the
   [releases page](https://github.com/OpenDroneMap/ODM/releases) and take the newest tag
   that actually has an `ODM_Setup_<version>.exe` asset. If it misbehaves, the pinned
   3.6.1 above is the known-good fallback.
3. Run the installer and **accept the default install location, `C:\ODM`.**

   > **Do not install into a path with spaces in it** (not `C:\Program Files\ODM`, not a
   > folder under `C:\Users\Your Name\`). The NodeODM Windows bundle you install next
   > predates its own fixes for quoting such paths, and it will fail in confusing ways.
   > `C:\ODM` also keeps project paths short, which matters because Windows limits paths
   > to 260 characters by default.

   > **Do not decline the UAC prompt.** The installer places ODM in `C:\ODM` as the
   > current user, but it also installs the Microsoft Visual C++ Redistributable, and
   > that part needs elevation. ODM 3.6.1 is built with the Visual Studio 2022 toolset
   > and its GDAL binaries crash on load against an older runtime - which produces a
   > task that fails instantly with a bare `Processing failed (3221225477)` and no log
   > output at all. See
   > [Troubleshooting](#troubleshooting) if you hit that.

4. When it finishes you should have `C:\ODM\run.bat`. Slicer looks for exactly that
   file to confirm the installation is real.

5. Verify ODM can actually run before going further - this catches the runtime problem
   above immediately:

   ```powershell
   cmd /c "pushd C:\ODM && call .\win32env.bat && gdalinfo --version"
   ```

   It should print something like `GDAL 3.11.1 "Eganville", released 2025/06/25`. If it
   prints nothing at all, your VC++ runtime is too old.

You do not need to open the ODM Console. Slicer never calls ODM directly - NodeODM does.

---

## Step 4 - Install NodeODM

NodeODM is a small server that puts a web API in front of ODM. Slicer talks to it, and
it runs your jobs. The Windows bundle is self-contained - no Node.js install needed.

1. Download the Windows bundle, **NodeODM v2.2.3**:

   <https://github.com/OpenDroneMap/NodeODM/releases/download/v2.2.3/nodeodm-windows-x64.zip>

   17 MB, published 15 May 2024. Unlike ODM, there is nothing newer to move to: v2.2.3 is
   the **last NodeODM release that ships a Windows bundle at all**, so this link is not
   going stale - it is the end of the line. It being that old is why it needs a one-file
   patch to work with current ODM; see
   [Step 4b](#step-4b---patch-one-file-in-nodeodm-required) below.
2. Extract it to **`C:\NodeODM`**, again avoiding any path with spaces.

   You should end up with `C:\NodeODM\nodeodm.exe` alongside `helpers\` and `apps\`
   folders. Keep those next to the executable; NodeODM looks for them relative to its
   own folder and will not start properly without them.

3. Your reconstruction projects are stored in `C:\NodeODM\data\`, so make sure that
   drive has room.

**About the SmartScreen warning.** The v2.2.3 executable is not code-signed, so Windows
will likely show "Windows protected your PC" the first time it runs. Click **More info >
Run anyway**. You can verify what you downloaded by checking it came from the
`github.com/OpenDroneMap/NodeODM` releases page over HTTPS.

### Step 4b - Patch one file in NodeODM (required)

**Without this, NodeODM starts and immediately exits.** This is not optional, and it is
not something you did wrong.

The v2.2.3 bundle ships `C:\NodeODM\helpers\odmOptionsToJson.py`, the script NodeODM uses
to ask ODM what options it supports. That script begins with `import imp`. Python removed
the `imp` module in version 3.12, and ODM 3.6.1 bundles Python 3.12.9 - so the script dies
on its first line, NodeODM cannot read ODM's option list, and it quits. What you see in
Slicer is:

```
error: Cannot read list of options from ODM (from temporary file). Is ODM installed in C:\ODM?
```

followed by *"NodeODM was started but did not respond on port 3002."* The paths are fine;
the two programs are simply a Python version apart.

NodeODM fixed this upstream after the v2.2.3 bundle was built, but no newer Windows
bundle has been published. So replace that single file with the current version.

In PowerShell:

```powershell
Copy-Item C:\NodeODM\helpers\odmOptionsToJson.py C:\NodeODM\helpers\odmOptionsToJson.py.orig
Invoke-WebRequest -Uri https://raw.githubusercontent.com/OpenDroneMap/NodeODM/2dc1819b0b047e56529d7dd23182e853fa509077/helpers/odmOptionsToJson.py -OutFile C:\NodeODM\helpers\odmOptionsToJson.py
```

Or download
[odmOptionsToJson.py](https://raw.githubusercontent.com/OpenDroneMap/NodeODM/2dc1819b0b047e56529d7dd23182e853fa509077/helpers/odmOptionsToJson.py)
in a browser and save it over `C:\NodeODM\helpers\odmOptionsToJson.py`, keeping a copy of
the original first.

That URL is pinned to commit
[`2dc1819`](https://github.com/OpenDroneMap/NodeODM/commit/2dc1819b0b047e56529d7dd23182e853fa509077),
"Upgrade to Python 3.12", which is the change that removed `imp` - 2,043 bytes, the file
tested here. Pinning matters more than usual for this one: a `master` link would hand you
whatever that file becomes later, and it has to stay compatible with a NodeODM bundle
frozen in 2024.

The replacement does the same job using `importlib` instead of `imp`. Nothing else in the
bundle uses `imp`, so this one file is the whole fix.

---

## Step 5 - Point Slicer at both

1. Open Slicer and go to the **Reconstruct 3D Models with ODM** module.

   The first time you open it, Slicer puts up an **Install pyodm** dialog - the module
   needs that Python package to talk to NodeODM. Click **OK** and give it a few seconds;
   a confirmation box follows, which you also dismiss. This happens once. Until you
   answer the dialog the rest of Slicer is unresponsive, which is normal - it is a modal
   prompt, not a freeze.

2. Open the **Manage NodeODM (Install/Launch)** section. On Windows it shows two extra
   fields:
   - **NodeODM executable** - set to `C:\NodeODM\nodeodm.exe`
   - **ODM install folder** - set to `C:\ODM`

   If you used the default locations above, these are filled in for you already.

3. Both paths are remembered, so this is a one-time step.
4. Click **Launch NodeODM**.

The Console Log shows what NodeODM prints as it starts. Within a few seconds you should
see it report that it is up. Slicer waits for the node to actually answer before saying
so, so if it reports a problem, read the log - the message is from NodeODM itself.

**The firewall prompt.** The first launch will probably raise a Windows Defender
Firewall dialog. NodeODM only needs to talk to Slicer on the same machine, so allowing
it on **private networks** is enough; you do not need to allow public networks.

To confirm it is running, click the **Node Dashboard** link in the Launch WebODM Task
section. It opens `http://127.0.0.1:3002` in your browser and lists the node's tasks.

---

## Running a reconstruction

From here the workflow is identical to Linux. See the [ODM User Guide](ODM.md) for the
full description of the parameters. In short:

1. Set **Masked Images Folder** to your masked images (from PhotoMasking or
   VideoMasking).
2. Adjust parameters if you need to; the defaults are tuned for specimen photography.
3. Click **Run NodeODM Task With Selected Parameters**. Uploading takes a few minutes
   for a few hundred images.
4. Watch the Console Log. When the task completes, results download automatically into a
   `WebODM_<name>` folder next to your images.
5. Click **Import Reconstructed Model** to load the textured mesh into Slicer.

**About the GPU.** If you have an NVIDIA driver installed, ODM finds it on its own -
there is nothing to enable. Look for `CUDA drivers detected` in the log. The point cloud
densification stage, which dominates the runtime, is the part that uses it. Leave the
`no-gpu` parameter set to `False`.

If you have no NVIDIA card, ODM logs `No CUDA drivers detected` and continues on the
CPU. Nothing fails; it just takes longer.

**Closing Slicer does not stop a running job.** NodeODM keeps working. Use
**Reconnect to Task on Node...** in a later session to pick the job back up and download
its results. Use **Stop Node** to shut NodeODM down when you are done.

---

## PhotoMasking and VideoMasking

**PhotoMasking** and **ClusterPhotos** need no Windows-specific setup. They install
their own Python packages into Slicer the first time you open them, and download model
weights on demand. Expect the first run to take a while and to need a few GB of disk.

Unlike ODM - which needs nothing but your GPU driver - the masking modules run PyTorch
themselves, and they install a **cu128** build of it through the PyTorch extension. This
is the one place where the CUDA build matters, and where a mismatch against your driver
shows up as a masking failure rather than an install error. See
[Troubleshooting](#troubleshooting) if a model fails to load on the GPU.

**VideoMasking** works the same way but has more moving parts. On first setup it:

- Fetches the SAMURAI source. It uses `git` when you have it, and downloads a zip when
  you do not - so you do not need to install git.
- Installs SAM 2 and its dependencies into Slicer's Python.
- Downloads the `sam2.1_hiera_large.pt` checkpoint (about 900 MB). Slicer may look
  frozen during this; give it time.

`ffmpeg` is **not** installed by Configure SAMURAI. The first time you extract frames
from a video, VideoMasking looks for `ffmpeg` on your PATH and, finding none, installs
`imageio-ffmpeg` and uses the binary bundled in that package. So expect one more short
download at the start of your first video run rather than during setup. You do not need
to install ffmpeg yourself either way.

Messages during setup that are expected on Windows and are not errors:

- Anything about the SAM 2 CUDA extension not being built. It is deliberately skipped;
  it only affects optional mask post-processing.
- `jpeg4py` installs without complaint but cannot actually load - the first call into it
  raises `OSError: Could not load libjpeg-turbo library`, because the pip package is a
  ctypes wrapper and the DLL it wants is not shipped for Windows. Nothing in the tracking
  pipeline uses it, so this is harmless wherever it surfaces.

---

## Troubleshooting

**"Set 'NodeODM executable' to your nodeodm.exe first"**
The path field is empty or points at a file that is not there. Redo
[Step 5](#step-5---point-slicer-at-both).

**"'ODM install folder' does not look like an ODM installation"**
The folder you chose has no `run.bat` in it. Point at the folder ODM_Setup installed
into - `C:\ODM` by default - not a subfolder of it.

**NodeODM starts and then exits a few seconds later**
Almost always a wrong **ODM install folder**. Read the Console Log; NodeODM says what it
could not find. Check that `C:\ODM\run.bat` exists.

**"NodeODM was started but did not respond on port 3002"**
First check the Console Log for `Cannot read list of options from ODM`. If it is there,
you skipped [Step 4b](#step-4b---patch-one-file-in-nodeodm-required) - NodeODM started,
failed to read ODM's options, and exited. This is the most common cause by far.

Otherwise something else may hold that port. Change **Node Port** in the Launch WebODM
Task section, then Stop Node and Launch again. A firewall block looks different: the
process keeps running and only the connection fails, so check whether `nodeodm.exe` is
still in Task Manager before blaming the firewall.

**A task fails instantly with "Processing failed (3221225477)"**
Your Microsoft Visual C++ Redistributable is too old. 3221225477 is `0xC0000005`, a
Windows access violation - ODM crashes on startup, before printing a single line, so the
Console Log and the node's task output are both empty.

ODM 3.6.1 is built with the Visual Studio 2022 toolset and needs the **14.4x** runtime.
If yours is older (14.22, from 2019, is a common one to be stuck on) every GDAL binary
inside ODM faults in `MSVCP140.dll` the moment it loads. Nothing about your images,
masks, or NodeODM is involved - `gdalinfo --version` crashes just as reliably.

Check what you have:

```powershell
(Get-Item C:\Windows\System32\MSVCP140.dll).VersionInfo.FileVersion
```

If that is below 14.40, install the current
[VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and reboot. This
is the step ODM_Setup tries to perform itself, so if you declined the UAC prompt during
installation, this is the result.

To confirm the fix before resubmitting a task:

```powershell
cmd /c "pushd C:\ODM && call win32env.bat && gdalinfo --version"
```

That must print a GDAL version. If it prints nothing, the runtime is still wrong.

**The installer asks for administrator rights**
ODM_Setup installs to `C:\ODM` as the current user, but it also installs the Microsoft
Visual C++ Redistributable if your machine does not already have it, and that part needs
elevation. If you cannot elevate, install the
[VC++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) separately first
via your IT department, then run ODM_Setup.

**A task sits at "Queued, Progress: 0%"**
Another task is ahead of it on the node. The log says how many. Open the Node Dashboard
to see or cancel them.

**Reconstruction fails partway through with an out-of-memory error**
Lower `pc-quality` and `feature-quality`, or reduce `max-concurrency`. Each concurrent
process needs its own memory.

**Everything is very slow**
Check whether the log says `CUDA drivers detected`. If not and you do have an NVIDIA
card, update your GPU driver. Very new cards occasionally need a newer ODM than the one
you installed.

**"Error loading SAM model: CUDA error: CUDA-capable device(s) is/are busy or unavailable"**
This is PhotoMasking or VideoMasking, not ODM - masking runs PyTorch on your GPU, and the
wheel it installed cannot claim the device. The masking modules install a specific CUDA
build of PyTorch through the PyTorch extension; on this branch that is **cu128**. If the
build in Slicer's Python does not match what your driver will serve, torch still imports
and still reports `torch.cuda.is_available() == True` - that flag only proves the driver
loaded, not that a context can be created - and the failure surfaces later, when a model
is actually pushed to the GPU.

Check what you have from the Slicer Python console:

```python
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.randn(8, 8, device="cuda").sum())   # this is the real test
```

The first line passing but the second raising is exactly this problem. To reinstall
against a different backend, replacing whatever is there now:

```python
import PyTorchUtils
logic = PyTorchUtils.PyTorchUtilsLogic()
logic.uninstallTorch(askConfirmation=False)
logic.installTorch(askConfirmation=False, forceComputationBackend="cu128")
```

Restart Slicer afterwards. Note that VideoMasking gates on the CUDA version it targets
and will refuse to run against a different one, so if you deliberately move off cu128 you
have to change `VideoMasking.py` to match.

---

## What is different from Linux

| | Linux | Windows |
|---|---|---|
| Reconstruction engine | `opendronemap/nodeodm:gpu` Docker image | ODM + NodeODM installed natively |
| Installed by | The module, via `docker pull` | You, once, by hand |
| GPU requirement | NVIDIA driver + Container Toolkit | NVIDIA driver only |
| Project storage | The module's `Resources/WebODM` folder | `C:\NodeODM\data` |
| Extension install | Extensions Manager | Local clone + additional module paths |
| Everything after "Launch" | | identical |

The reconstruction itself is the same ODM pipeline producing the same outputs. Only how
the engine gets onto the machine differs.

---

## Credits

See the main [README](../README.md) for citation information and acknowledgements. ODM
and NodeODM are projects of [OpenDroneMap](https://opendronemap.org).

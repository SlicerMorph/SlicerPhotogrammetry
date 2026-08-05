# Running Photogrammetry on Windows

The Photogrammetry extension was built for Linux, where the reconstruction engine runs
in a Docker container. On Windows there is no Docker in this workflow. Instead you
install two OpenDroneMap programs natively, once, and point Slicer at them.

Everything here is a manual, one-time setup. The extension never downloads or installs
these programs for you.

> **Status: this branch is not yet verified on Windows.** The `windows-only` branch
> exists so this can be tested. If a step below is wrong, please open an issue.

## Table of Contents

1. [What you need](#what-you-need)
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
- About **12 GB** of free disk space for the software, plus room for your projects.
  Reconstructions are large; budget several GB per dataset.
- **16 GB RAM minimum**, more is better. Photogrammetry is memory-hungry.
- An **NVIDIA GPU is optional but strongly recommended.** Only the driver is needed -
  no CUDA toolkit, and no NVIDIA Container Toolkit. If you have no NVIDIA card,
  everything still works on the CPU, just slower.
- A network connection for the downloads below.

You do **not** need Docker, WSL, git, or admin rights on the Slicer side. The ODM
installer may ask for admin once (see [Troubleshooting](#troubleshooting)).

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

1. Go to the [ODM releases page](https://github.com/OpenDroneMap/ODM/releases).
2. Download the `ODM_Setup_<version>.exe` asset from the latest release that has one.
   As of August 2026 that is **ODM_Setup_3.6.1.exe** (about 245 MB).

   Note that not every ODM release ships a Windows installer - if the newest release has
   only source archives, take the most recent one that has an `.exe`.
3. Run the installer and **accept the default install location, `C:\ODM`.**

   > **Do not install into a path with spaces in it** (not `C:\Program Files\ODM`, not a
   > folder under `C:\Users\Your Name\`). The NodeODM Windows bundle you install next
   > predates its own fixes for quoting such paths, and it will fail in confusing ways.
   > `C:\ODM` also keeps project paths short, which matters because Windows limits paths
   > to 260 characters by default.

4. When it finishes you should have `C:\ODM\run.bat`. Slicer looks for exactly that
   file to confirm the installation is real.

You do not need to open the ODM Console. Slicer never calls ODM directly - NodeODM does.

---

## Step 4 - Install NodeODM

NodeODM is a small server that puts a web API in front of ODM. Slicer talks to it, and
it runs your jobs. The Windows bundle is self-contained - no Node.js install needed.

1. Go to the [NodeODM releases page](https://github.com/OpenDroneMap/NodeODM/releases).
2. Download **`nodeodm-windows-x64.zip`** (about 18 MB) from the latest release that has
   it - currently **v2.2.3**.
3. Extract it to **`C:\NodeODM`**, again avoiding any path with spaces.

   You should end up with `C:\NodeODM\nodeodm.exe` alongside `helpers\` and `apps\`
   folders. Keep those next to the executable; NodeODM looks for them relative to its
   own folder and will not start properly without them.

4. Your reconstruction projects are stored in `C:\NodeODM\data\`, so make sure that
   drive has room.

**About the SmartScreen warning.** The v2.2.3 executable is not code-signed, so Windows
will likely show "Windows protected your PC" the first time it runs. Click **More info >
Run anyway**. You can verify what you downloaded by checking it came from the
`github.com/OpenDroneMap/NodeODM` releases page over HTTPS.

---

## Step 5 - Point Slicer at both

1. Open Slicer and go to the **Reconstruct 3D Models with ODM** module.
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

**VideoMasking** works the same way but has more moving parts. On first setup it:

- Fetches the SAMURAI source. It uses `git` when you have it, and downloads a zip when
  you do not - so you do not need to install git.
- Installs SAM 2 and its dependencies into Slicer's Python.
- Downloads the `sam2.1_hiera_large.pt` checkpoint (about 900 MB). Slicer may look
  frozen during this; give it time.
- Installs `imageio-ffmpeg` if there is no `ffmpeg` on your PATH, which provides the
  ffmpeg used for frame extraction. You do not need to install ffmpeg yourself.

Two messages during setup are expected on Windows and are not errors:

- `optional package could not be installed: jpeg4py` - that package has no working
  Windows build and nothing in the tracking pipeline uses it.
- Anything about the SAM 2 CUDA extension not being built. It is deliberately skipped;
  it only affects optional mask post-processing.

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
Something else may hold that port. Change **Node Port** in the Launch WebODM Task
section, then Stop Node and Launch again. Otherwise read the Console Log.

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

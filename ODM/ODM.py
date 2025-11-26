#
# ODM.py
#
# ODM (OpenDroneMap/NodeODM) module for 3D Slicer
# Extracted from PhotoMasking module for better modularity
#

import os
import sys
import qt
import ctk
import slicer
import subprocess
import json
from slicer.ScriptedLoadableModule import *


class ODM(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "Reconstruct 3D Models with ODM"
        self.parent.categories = ["SlicerMorph.Photogrammetry"]
        self.parent.dependencies = []
        self.parent.contributors = ["Oshane Thomas (SCRI), Murat Maga (SCRI)"]
        self.parent.helpText = """ODM is a 3D Slicer module for managing NodeODM/WebODM 
        photogrammetry reconstruction tasks. This module provides tools to launch NodeODM, 
        configure reconstruction parameters, monitor task progress, and import the final 3D 
        models into Slicer. It accepts masked images from either PhotoMasking or VideoMasking modules."""
        
        self.parent.acknowledgementText = """This module was developed with support from the National Science 
        Foundation under grants DBI/2301405 and OAC/2118240 awarded to AMM at Seattle Children's Research Institute."""


class ODMWidget(ScriptedLoadableModuleWidget):
    """
    UI and logic for the ODM module.
    Manages:
     - NodeODM installation and launching (Docker-based)
     - Input folder selection (for masked images from PhotoMasking or VideoMasking)
     - WebODM task configuration and execution
     - Task monitoring and result downloading
     - 3D model import into Slicer
     - Task save/restore functionality
    """

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        
        # Input folder
        self.inputFolderSelector = None
        
        # NodeODM connection
        self.nodeIPLineEdit = None
        self.nodePortSpinBox = None
        
        # WebODM task UI
        self.launchWebODMTaskButton = None
        self.webodmLogTextEdit = None
        self.stopMonitoringButton = None
        
        # NodeODM management
        self.launchWebODMButton = None
        self.stopWebODMButton = None
        
        # Model import
        self.importModelButton = None
        
        # Save/Restore
        self.saveTaskButton = None
        self.restoreTaskButton = None
        
        # WebODM baseline parameters
        self.baselineParams = {
            "orthophoto-resolution": 0.3,
            "skip-orthophoto": True,
            "texturing-single-material": True,
            "use-3dmesh": True,
        }
        
        # WebODM parameter levels
        self.factorLevels = {
            "ignore-gsd": [False, True],
            "matcher-neighbors": [16, 0, 8, 10, 12, 24],
            "mesh-octree-depth": [12, 13, 14],
            "mesh-size": [300000, 500000, 750000, 1000000],
            "min-num-features": [50000, 10000, 20000],
            "pc-filter": [1, 2, 3, 4, 5],
            "depthmap-resolution": [3072, 2048, 4096, 8192],
            "matcher-type": ["bruteforce", "bow", "flann"],
            "feature-type": ["dspsift", "akaze", "hahog", "orb", "sift"],
            "feature-quality": ["ultra", "medium", "high"],
            "pc-quality": ["high", "medium", "ultra"],
            "optimize-disk-space": [True, False],
            "rerun": ["openmvs", "dataset", "split", "merge", "opensfm"],
            "no-gpu": [False, True],
        }
        self.factorComboBoxes = {}
        
        # Dataset name and concurrency
        self.datasetNameLineEdit = None
        self.maxConcurrencySpinBox = None
        
        # GCP (Ground Control Points)
        self.findGCPScriptSelector = None
        self.generateGCPButton = None
        self.gcpCoordFileSelector = None
        self.arucoDictIDSpinBox = None
        self.gcpListContent = ""
        self.gcpCoordFilePath = ""
        
        # "Clone Find-GCP" button
        self.cloneFindGCPButton = None
        
        # Manager for WebODM operations
        self.webODMManager = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        
        # Check and install pyodm if needed
        self._ensurePyODMInstalled()
        
        #
        # Input Folder Selection
        #
        inputCollapsible = ctk.ctkCollapsibleButton()
        inputCollapsible.text = "Masked Images Folder"
        self.layout.addWidget(inputCollapsible)
        inputFormLayout = qt.QFormLayout(inputCollapsible)
        
        self.inputFolderSelector = ctk.ctkDirectoryButton()
        self.inputFolderSelector.setToolTip("Select the folder containing masked images from PhotoMasking or VideoMasking")
        inputFormLayout.addRow("Masked Images Folder:", self.inputFolderSelector)
        
        #
        # Manage WebODM (Install/Launch) Collapsible
        #
        manageWODMCollapsible = ctk.ctkCollapsibleButton()
        manageWODMCollapsible.text = "Manage NodeODM (Install/Launch)"
        self.layout.addWidget(manageWODMCollapsible)
        manageWODMFormLayout = qt.QFormLayout(manageWODMCollapsible)
        
        buttonRow = qt.QHBoxLayout()
        self.launchWebODMButton = qt.QPushButton("Launch NodeODM")
        self.stopWebODMButton = qt.QPushButton("Stop Node")
        buttonRow.addWidget(self.launchWebODMButton)
        buttonRow.addWidget(self.stopWebODMButton)
        manageWODMFormLayout.addRow(buttonRow)
        
        #
        # Find-GCP Collapsible
        #
        findGCPCollapsible = ctk.ctkCollapsibleButton()
        findGCPCollapsible.text = "Find-GCP"
        self.layout.addWidget(findGCPCollapsible)
        findGCPFormLayout = qt.QFormLayout(findGCPCollapsible)
        
        self.cloneFindGCPButton = qt.QPushButton("Clone Find-GCP")
        findGCPFormLayout.addWidget(self.cloneFindGCPButton)
        self.cloneFindGCPButton.connect('clicked(bool)', self.onCloneFindGCPClicked)
        
        self.findGCPScriptSelector = ctk.ctkPathLineEdit()
        self.findGCPScriptSelector.filters = ctk.ctkPathLineEdit().Files
        self.findGCPScriptSelector.setToolTip("Select path to Find-GCP.py script.")
        findGCPFormLayout.addRow("Find-GCP Script:", self.findGCPScriptSelector)
        
        savedFindGCPScript = slicer.app.settings().value("ODM/findGCPScriptPath", "")
        if os.path.isfile(savedFindGCPScript):
            self.findGCPScriptSelector.setCurrentPath(savedFindGCPScript)
        self.findGCPScriptSelector.connect('currentPathChanged(QString)', self.onFindGCPScriptChanged)
        
        self.gcpCoordFileSelector = ctk.ctkPathLineEdit()
        self.gcpCoordFileSelector.filters = ctk.ctkPathLineEdit().Files
        self.gcpCoordFileSelector.setToolTip("Select GCP coordinate file (required).")
        findGCPFormLayout.addRow("GCP Coord File:", self.gcpCoordFileSelector)
        
        self.arucoDictIDSpinBox = qt.QSpinBox()
        self.arucoDictIDSpinBox.setMinimum(0)
        self.arucoDictIDSpinBox.setMaximum(99)
        self.arucoDictIDSpinBox.setValue(2)
        findGCPFormLayout.addRow("ArUco Dictionary ID:", self.arucoDictIDSpinBox)
        
        self.generateGCPButton = qt.QPushButton("Generate GCP File from Images")
        findGCPFormLayout.addWidget(self.generateGCPButton)
        self.generateGCPButton.connect('clicked(bool)', self.onGenerateGCPClicked)
        self.generateGCPButton.setEnabled(True)
        
        #
        # Launch WebODM Task Collapsible
        #
        webodmTaskCollapsible = ctk.ctkCollapsibleButton()
        webodmTaskCollapsible.text = "Launch WebODM Task"
        self.layout.addWidget(webodmTaskCollapsible)
        webodmTaskFormLayout = qt.QFormLayout(webodmTaskCollapsible)
        
        self.nodeIPLineEdit = qt.QLineEdit("127.0.0.1")
        self.nodeIPLineEdit.setToolTip("Enter the IP address of the NodeODM instance (e.g. 127.0.0.1).")
        webodmTaskFormLayout.addRow("Node IP:", self.nodeIPLineEdit)
        
        self.nodePortSpinBox = qt.QSpinBox()
        self.nodePortSpinBox.setMinimum(1)
        self.nodePortSpinBox.setMaximum(65535)
        self.nodePortSpinBox.setValue(3002)
        self.nodePortSpinBox.setToolTip("Port number on which NodeODM is listening. Commonly 3001 or 3002.")
        webodmTaskFormLayout.addRow("Node Port:", self.nodePortSpinBox)
        
        # WebODM parameter tooltips
        parameterTooltips = {
            "ignore-gsd": (
                "Ignore Ground Sampling Distance (GSD). A memory/processor-hungry setting if true.\n"
                "Ordinarily, GSD caps maximum resolution. Use with caution.\nDefault: False"
            ),
            "matcher-neighbors": (
                "Perform image matching with the nearest images based on GPS exif data.\n"
                "Set to 0 to match by triangulation.\nDefault: 0"
            ),
            "mesh-octree-depth": (
                "Octree depth used in mesh reconstruction. Increase for more vertices.\n"
                "Typical range 8-12.\nDefault: 11"
            ),
            "mesh-size": (
                "Max vertex count for the output mesh.\nDefault: 200000"
            ),
            "min-num-features": (
                "Minimum number of features to extract per image.\n"
                "Higher values can help with low-overlap areas but increase processing.\nDefault: 10000"
            ),
            "pc-filter": (
                "Filters the point cloud by removing outliers.\n"
                "Value = # of standard deviations from local mean.\nDefault: 5"
            ),
            "depthmap-resolution": (
                "Sets the resolution for depth maps.\n"
                "Higher values = more detail, but more memory/time.\nTypical range 2048..8192.\nDefault: 2048"
            ),
            "matcher-type": (
                "Matcher algorithm: bruteforce, bow, or flann.\n"
                "FLANN is slower but stable, BOW is faster but might miss matches,\n"
                "BRUTEFORCE is slow but robust.\nDefault: flann"
            ),
            "feature-type": (
                "Keypoint/descriptor algorithm: akaze, dspsift, hahog, orb, sift.\n"
                "Default: dspsift"
            ),
            "feature-quality": (
                "Feature extraction quality: ultra, high, medium, low, lowest.\n"
                "Higher quality = better features, but slower.\nDefault: high"
            ),
            "pc-quality": (
                "Point cloud quality: ultra, high, medium, low, lowest.\n"
                "Higher = denser cloud, more resources.\nDefault: medium"
            ),
            "optimize-disk-space": (
                "Delete large intermediate files to reduce disk usage.\n"
                "Prevents partial pipeline restarts.\nDefault: False"
            ),
            "rerun": (
                "Rerun only a specific pipeline stage and stop.\n"
                "Options: dataset, split, merge, opensfm, openmvs, etc.\n"
                "Default: (none)"
            ),
            "no-gpu": (
                "Disable GPU usage even if available.\nDefault: False"
            ),
        }
        
        # Create combo boxes for each WebODM parameter
        for factorName, levels in self.factorLevels.items():
            combo = qt.QComboBox()
            for val in levels:
                combo.addItem(str(val))
            
            # Assign tooltip
            if factorName in parameterTooltips:
                combo.setToolTip(parameterTooltips[factorName])
            else:
                combo.setToolTip(f"Parameter '{factorName}' is not documented in the tooltips dictionary.")
            
            self.factorComboBoxes[factorName] = combo
            webodmTaskFormLayout.addRow(f"{factorName}:", combo)
        
        # Max concurrency
        self.maxConcurrencySpinBox = qt.QSpinBox()
        self.maxConcurrencySpinBox.setRange(16, 256)
        self.maxConcurrencySpinBox.setValue(16)
        self.maxConcurrencySpinBox.setToolTip(
            "Maximum number of processes used by WebODM.\n"
            "Higher values = faster but more memory usage."
        )
        webodmTaskFormLayout.addRow("max-concurrency:", self.maxConcurrencySpinBox)
        
        # TODO: Add WebODM parameter controls here (will be added in next step)
        
        self.datasetNameLineEdit = qt.QLineEdit("SlicerReconstruction")
        self.datasetNameLineEdit.setToolTip("Name of the dataset in WebODM.\\nThis will be the reconstruction folder label.")
        webodmTaskFormLayout.addRow("name:", self.datasetNameLineEdit)
        
        self.launchWebODMTaskButton = qt.QPushButton("Run NodeODM Task With Selected Parameters (non-blocking)")
        webodmTaskFormLayout.addWidget(self.launchWebODMTaskButton)
        self.launchWebODMTaskButton.setEnabled(True)
        
        self.webodmLogTextEdit = qt.QTextEdit()
        self.webodmLogTextEdit.setReadOnly(True)
        webodmTaskFormLayout.addRow("Console Log:", self.webodmLogTextEdit)
        
        self.stopMonitoringButton = qt.QPushButton("Stop Monitoring")
        self.stopMonitoringButton.setEnabled(False)
        webodmTaskFormLayout.addWidget(self.stopMonitoringButton)
        
        self.importModelButton = qt.QPushButton("Import Reconstructed Model")
        self.layout.addWidget(self.importModelButton)
        
        # #
        # # Save/Restore Task (TODO: Future feature)
        # #
        # saveRestoreCollapsible = ctk.ctkCollapsibleButton()
        # saveRestoreCollapsible.text = "Save/Restore Task"
        # self.layout.addWidget(saveRestoreCollapsible)
        # saveRestoreLayout = qt.QFormLayout(saveRestoreCollapsible)
        # 
        # self.saveTaskButton = qt.QPushButton("Save Task")
        # self.restoreTaskButton = qt.QPushButton("Restore Task")
        # buttonsRow = qt.QHBoxLayout()
        # buttonsRow.addWidget(self.saveTaskButton)
        # self.saveTaskButton.enabled = False
        # self.restoreTaskButton.enabled = False
        # buttonsRow.addWidget(self.restoreTaskButton)
        # saveRestoreLayout.addRow(buttonsRow)
        
        # Add stretch
        self.layout.addStretch(1)
        
        # Connect signals
        self.launchWebODMButton.connect('clicked(bool)', self.onLaunchWebODMClicked)
        self.stopWebODMButton.connect('clicked(bool)', self.onStopNodeClicked)
        self.launchWebODMTaskButton.connect('clicked(bool)', self.onRunWebODMTask)
        self.stopMonitoringButton.connect('clicked(bool)', self.onStopMonitoring)
        self.importModelButton.connect('clicked(bool)', self.onImportModelClicked)
        # self.saveTaskButton.connect('clicked(bool)', self.onSaveTaskClicked)
        # self.restoreTaskButton.connect('clicked(bool)', self.onRestoreTaskClicked)
        
        # Setup WebODM local folder
        modulePath = os.path.dirname(slicer.modules.odm.path)
        self.webODMLocalFolder = os.path.join(modulePath, 'Resources', 'WebODM')
        self.ensure_webodm_folder_permissions()
        
        # Initialize WebODM manager
        self.webODMManager = ODMManager(widget=self)

    def ensure_webodm_folder_permissions(self):
        """Ensure the WebODM folder exists with proper permissions."""
        import stat
        import logging
        
        try:
            if not os.path.exists(self.webODMLocalFolder):
                os.makedirs(self.webODMLocalFolder, exist_ok=True)
            
            # Set permissions: 0777 (rwxrwxrwx) so Docker container can write
            # This is necessary because NodeODM runs as a different user inside the container
            os.chmod(self.webODMLocalFolder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            logging.info(f"WebODM folder created and permissions set: {self.webODMLocalFolder}")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to create or set permissions for WebODM folder:\\n{str(e)}")
    
    def _ensurePyODMInstalled(self):
        """Check if pyodm is installed, and install it if missing."""
        try:
            import pyodm  # noqa: F401
            # Already installed
            return
        except ImportError:
            pass
        
        # Ask user to install
        if not slicer.util.confirmOkCancelDisplay(
            "The ODM module requires the 'pyodm' Python package.\\n\\n"
            "Install it now?",
            "Install pyodm"
        ):
            slicer.util.warningDisplay(
                "pyodm is required for this module to function.\\n"
                "You can install it manually via:\\n"
                "pip install pyodm"
            )
            return
        
        try:
            slicer.util.pip_install("pyodm")
            slicer.util.infoDisplay("pyodm installed successfully!")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to install pyodm:\\n{e}\\n\\nPlease install manually:\\npip install pyodm")

    def onLaunchWebODMClicked(self):
        """Launch NodeODM container with GPU support on port 3002"""
        self.webODMManager.onLaunchWebODMClicked()
    
    def onStopNodeClicked(self):
        """Stop the running NodeODM container"""
        self.webODMManager.onStopNodeClicked()
    
    def onRunWebODMTask(self):
        """Create and run a WebODM reconstruction task"""
        self.webODMManager.onRunWebODMTask()
    
    def onStopMonitoring(self):
        """Stop monitoring the current task"""
        self.webODMManager.onStopMonitoring()
    
    def onImportModelClicked(self):
        """Import the reconstructed 3D model into Slicer"""
        self.webODMManager.onImportModelClicked()
    
    # def onSaveTaskClicked(self):
    #     """Save the current task configuration to JSON"""
    #     # TODO: Implement save functionality
    #     slicer.util.infoDisplay("Save task functionality coming soon")
    # 
    # def onRestoreTaskClicked(self):
    #     """Restore a previously saved task"""
    #     # TODO: Implement restore functionality
    #     slicer.util.infoDisplay("Restore task functionality coming soon")
    
    def onFindGCPScriptChanged(self, newPath):
        """Save the Find-GCP script path to settings."""
        if os.path.isfile(newPath):
            slicer.app.settings().setValue("ODM/findGCPScriptPath", newPath)
    
    def onGenerateGCPClicked(self):
        """Generate a combined GCP list file from all images in the input folder."""
        import subprocess
        import hashlib
        
        find_gcp_script = self.findGCPScriptSelector.currentPath
        if not find_gcp_script or not os.path.isfile(find_gcp_script):
            slicer.util.errorDisplay("Please select a valid Find-GCP.py script path.")
            return
        
        self.gcpCoordFilePath = self.gcpCoordFileSelector.currentPath
        if not self.gcpCoordFilePath or not os.path.isfile(self.gcpCoordFilePath):
            slicer.util.errorDisplay("Please select a valid GCP coordinate file (required).")
            return
        
        inputFolder = self.inputFolderSelector.directory
        if not inputFolder or not os.path.isdir(inputFolder):
            slicer.util.errorDisplay("Please select a valid input folder containing masked images.")
            return
        
        # Output GCP file will be placed in the input folder
        combinedOutputFile = os.path.join(inputFolder, "combined_gcp_list.txt")
        
        # Collect all image files from input folder
        imageExtensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")
        allImages = []
        for filename in os.listdir(inputFolder):
            lower_fn = filename.lower()
            # Exclude mask files (case-insensitive)
            if filename.endswith(imageExtensions) and not lower_fn.endswith("_mask.jpg") and not lower_fn.endswith("_mask.jpeg") and not lower_fn.endswith("_mask.png"):
                allImages.append(os.path.join(inputFolder, filename))
        
        if len(allImages) == 0:
            slicer.util.warningDisplay("No images found in input folder. Nothing to do.")
            return
        
        dict_id = self.arucoDictIDSpinBox.value
        cmd = [
            sys.executable,
            find_gcp_script,
            "-t", "ODM",
            "-d", str(dict_id),
            "-i", self.gcpCoordFilePath,
            "--epsg", "3857",
            "-o", combinedOutputFile
        ]
        cmd += allImages
        
        try:
            slicer.util.infoDisplay("Running Find-GCP to produce a combined gcp_list.txt...")
            subprocess.run(cmd, check=True)
            
            if os.path.isfile(combinedOutputFile):
                with open(combinedOutputFile, "r") as f:
                    self.gcpListContent = f.read()
                
                slicer.util.infoDisplay(
                    f"Combined GCP list created successfully at:\n{combinedOutputFile}",
                    autoCloseMsec=3500
                )
            else:
                slicer.util.warningDisplay(f"Find-GCP did not produce the file:\n{combinedOutputFile}")
        
        except subprocess.CalledProcessError as e:
            slicer.util.warningDisplay(f"Find-GCP failed (CalledProcessError): {str(e)}")
        except Exception as e:
            slicer.util.warningDisplay(f"An error occurred running Find-GCP: {str(e)}")
    
    def onCloneFindGCPClicked(self):
        """Clone the Find-GCP repository from GitHub."""
        import shutil
        from slicer.util import downloadFile, extractArchive
        
        # Paths we will use:
        modulePath = os.path.dirname(slicer.modules.odm.path)
        resourcesFolder = os.path.join(modulePath, "Resources")
        os.makedirs(resourcesFolder, exist_ok=True)
        
        # 1) Where to save the downloaded .zip
        zipFilePath = os.path.join(resourcesFolder, "Find-GCP.zip")
        # 2) The name of the folder that GitHub's master.zip will produce
        extractedFolderName = "Find-GCP-master"
        # 3) Full path to that folder after extraction
        cloneFolder = os.path.join(resourcesFolder, extractedFolderName)
        # 4) The script we expect inside that extracted folder
        scriptInsideClone = os.path.join(cloneFolder, "gcp_find.py")
        
        # If script already exists, just set the path and return
        if os.path.isfile(scriptInsideClone):
            self.findGCPScriptSelector.setCurrentPath(scriptInsideClone)
            slicer.app.settings().setValue("ODM/findGCPScriptPath", scriptInsideClone)
            slicer.util.infoDisplay(
                f"Find-GCP already exists at:\n{scriptInsideClone}",
                autoCloseMsec=2000
            )
            return
        
        # Direct download link for the .zip (refs/heads/master)
        url = "https://github.com/SlicerMorph/Find-GCP/archive/refs/heads/master.zip"
        
        # Show downloading message (no auto-close, will update when done)
        progressDialog = slicer.util.createProgressDialog(
            labelText="Downloading Find-GCP from GitHub...",
            maximum=0
        )
        progressDialog.show()
        slicer.app.processEvents()
        
        try:
            downloadFile(url, zipFilePath)
        except Exception as e:
            progressDialog.close()
            slicer.util.errorDisplay(f"Failed to download Find-GCP zip:\n{str(e)}")
            return
        
        # Update progress
        progressDialog.labelText = "Extracting Find-GCP..."
        slicer.app.processEvents()
        
        # Remove old folder if it exists
        if os.path.isdir(cloneFolder):
            shutil.rmtree(cloneFolder)
        
        # Extract the .zip
        try:
            extractArchive(zipFilePath, resourcesFolder)
        except Exception as e:
            progressDialog.close()
            slicer.util.errorDisplay(f"Failed to extract Find-GCP: {str(e)}")
            return
        
        progressDialog.close()
        
        # Verify the script exists
        if os.path.isfile(scriptInsideClone):
            self.findGCPScriptSelector.setCurrentPath(scriptInsideClone)
            slicer.app.settings().setValue("ODM/findGCPScriptPath", scriptInsideClone)
            slicer.util.infoDisplay(
                f"Find-GCP ready at:\n{scriptInsideClone}",
                autoCloseMsec=3000
            )
        else:
            slicer.util.warningDisplay(f"gcp_find.py not found after extraction in:\n{cloneFolder}")


class ODMManager:
    """
    Manager class dedicated to WebODM/NodeODM functionality:
     - Checking Docker / WebODM status
     - Installing / Re-installing WebODM
     - Launching a container with GPU support on port 3002
     - Stopping a running node
     - Creating / monitoring a pyodm Task
     - Downloading results on completion
     - Stopping task monitoring
     - Importing the completed model into Slicer
    """

    def __init__(self, widget):
        self.widget = widget
        self.webodmTask = None
        self.webodmOutDir = None
        self.webodmTimer = None
        self.lastWebODMOutputLineIndex = 0

    def onLaunchWebODMClicked(self):
        """
        Launch NodeODM container with GPU support on port 3002
        """
        proceed = slicer.util.confirmYesNoDisplay(
            "This action will ensure nodeodm:gpu is installed (pull if needed), "
            "stop any running container on port 3002, and launch a new one.\\n\\n"
            "Proceed?"
        )
        if not proceed:
            slicer.util.infoDisplay("Launch NodeODM canceled by user.")
            return
        
        # Check Docker is available
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
        except Exception as e:
            slicer.util.warningDisplay(f"Docker not found or not in PATH.\\nError: {str(e)}")
            return
        
        # Check if image exists, pull if needed
        try:
            check_process = subprocess.run(
                ["docker", "images", "-q", "opendronemap/nodeodm:gpu"],
                capture_output=True,
                text=True,
                check=True
            )
            image_id = check_process.stdout.strip()
            if not image_id:
                slicer.util.infoDisplay("nodeodm:gpu not found, pulling latest (this may take a while).")
                pull_process = subprocess.run(
                    ["docker", "pull", "opendronemap/nodeodm:gpu"],
                    text=True
                )
                if pull_process.returncode != 0:
                    slicer.util.errorDisplay("Failed to pull nodeodm:gpu image. Check logs.")
                    return
                else:
                    slicer.util.infoDisplay("Successfully pulled nodeodm:gpu.")
        except subprocess.CalledProcessError as e:
            slicer.util.errorDisplay(f"Error checking nodeodm:gpu status: {str(e)}")
            return
        
        # Stop any existing containers on port 3002
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "publish=3002", "--format", "{{.ID}}"],
                capture_output=True, text=True, check=True
            )
            container_ids = result.stdout.strip().split()
            for cid in container_ids:
                if cid:
                    slicer.util.infoDisplay(f"Stopping container {cid} on port 3002...")
                    subprocess.run(["docker", "stop", cid], check=True)
        except Exception as e:
            slicer.util.warningDisplay(f"Error stopping old container(s): {str(e)}")
        
        # Ensure local folder exists
        local_folder = self.widget.webODMLocalFolder
        if not os.path.isdir(local_folder):
            slicer.util.infoDisplay("Creating local WebODM folder...")
            os.makedirs(local_folder, exist_ok=True)
        
        # Launch container with volume mount
        slicer.util.infoDisplay("Launching nodeodm:gpu container on port 3002...")
        cmd = [
            "docker", "run", "--rm", "-d",
            "-p", "3002:3000",
            "--gpus", "all",
            "--name", "slicer-webodm-3002",
            "-v", f"{local_folder}:/var/www/data",
            "opendronemap/nodeodm:gpu"
        ]
        try:
            subprocess.run(cmd, check=True)
            slicer.util.infoDisplay("WebODM launched successfully on port 3002.")
            self.widget.nodeIPLineEdit.setText("127.0.0.1")
            self.widget.nodePortSpinBox.setValue(3002)
            slicer.app.settings().setValue("ODM/WebODMIP", "127.0.0.1")
            slicer.app.settings().setValue("ODM/WebODMPort", "3002")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to launch WebODM container:\\n{str(e)}")

    def onStopNodeClicked(self):
        """
        Stop the running NodeODM container on port 3002
        """
        jobInProgress = (self.webodmTask is not None)

        if jobInProgress:
            proceed = slicer.util.confirmYesNoDisplay(
                "A WebODM task appears to be in progress. Stopping the node now will cancel that task.\\n\\n"
                "Do you want to continue?"
            )
            if not proceed:
                slicer.util.infoDisplay("Stop Node canceled by user.")
                return

        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "publish=3002", "--format", "{{.ID}}"],
                capture_output=True, text=True, check=True
            )
            container_ids = result.stdout.strip().split()
            if not container_ids or not any(container_ids):
                slicer.util.infoDisplay("No container currently running on port 3002.")
                return

            for cid in container_ids:
                if cid:
                    slicer.util.infoDisplay(f"Stopping container {cid} on port 3002...")
                    subprocess.run(["docker", "stop", cid], check=True)
            slicer.util.infoDisplay("NodeODM container stopped.")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to stop container:\\n{str(e)}")

    def onRunWebODMTask(self):
        """
        Create and execute a WebODM reconstruction task
        """
        try:
            from pyodm import Node
        except ImportError:
            slicer.util.errorDisplay("pyodm module not found. Please install it via pip.")
            return

        node_ip = self.widget.nodeIPLineEdit.text.strip()
        node_port = self.widget.nodePortSpinBox.value
        try:
            node = Node(node_ip, node_port)
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to connect to Node at {node_ip}:{node_port}\\n{str(e)}")
            return

        inputFolder = self.widget.inputFolderSelector.directory
        if not inputFolder or not os.path.isdir(inputFolder):
            slicer.util.errorDisplay("Input folder is invalid. Please select a folder containing masked images.")
            return

        # Collect masked images and mask files (JPG and PNG)
        all_masked_color_images = []
        all_mask_images = []
        for root, dirs, files in os.walk(inputFolder):
            for fn in files:
                lower_fn = fn.lower()
                # Check for color images (JPG/JPEG or PNG, but not masks)
                if (lower_fn.endswith(".jpg") or lower_fn.endswith(".jpeg") or lower_fn.endswith(".png")) and not lower_fn.endswith("_mask.jpg") and not lower_fn.endswith("_mask.jpeg") and not lower_fn.endswith("_mask.png"):
                    all_masked_color_images.append(os.path.join(root, fn))
                # Check for mask images (JPG/JPEG or PNG)
                elif lower_fn.endswith("_mask.jpg") or lower_fn.endswith("_mask.jpeg") or lower_fn.endswith("_mask.png"):
                    all_mask_images.append(os.path.join(root, fn))

        all_images = all_masked_color_images + all_mask_images
        if len(all_images) == 0:
            slicer.util.warningDisplay("No masked images (JPG/JPEG or PNG) found in input folder.")
            return

        # Check for combined GCP file
        combinedGCP = os.path.join(inputFolder, "combined_gcp_list.txt")
        files_to_upload = all_images[:]
        if os.path.isfile(combinedGCP):
            files_to_upload.append(combinedGCP)
        else:
            slicer.util.infoDisplay("No combined_gcp_list.txt found. Proceeding without GCP...")

        # Build parameters from baseline + UI selections
        params = dict(self.widget.baselineParams)

        for factorName, combo in self.widget.factorComboBoxes.items():
            chosen_str = combo.currentText
            if factorName == "ignore-gsd":
                params["ignore-gsd"] = (chosen_str.lower() == "true")
            elif factorName == "optimize-disk-space":
                params["optimize-disk-space"] = (chosen_str.lower() == "true")
            elif factorName == "no-gpu":
                params["no-gpu"] = (chosen_str.lower() == "true")
            else:
                try:
                    val_int = int(chosen_str)
                    params[factorName] = val_int
                except ValueError:
                    params[factorName] = chosen_str

        params["max-concurrency"] = self.widget.maxConcurrencySpinBox.value
        
        # Generate task name based on parameters (creates a short hash-based name)
        prefix = self.widget.datasetNameLineEdit.text.strip() or "SlicerReconstruction"
        shortTaskName = self.generateShortTaskName(prefix, params)

        slicer.util.infoDisplay("Creating WebODM Task (non-blocking). Upload may take time...")

        try:
            self.webodmTask = node.create_task(files=files_to_upload, options=params, name=shortTaskName)
        except Exception as e:
            slicer.util.errorDisplay(f"Task creation failed:\\n{str(e)}")
            return

        slicer.util.infoDisplay(f"Task '{shortTaskName}' created successfully. Monitoring progress...")

        self.webodmOutDir = os.path.join(inputFolder, f"WebODM_{shortTaskName}")
        os.makedirs(self.webodmOutDir, exist_ok=True)

        self.widget.webodmLogTextEdit.clear()
        self.widget.stopMonitoringButton.setEnabled(True)

        self.lastWebODMOutputLineIndex = 0

        if self.webodmTimer:
            self.webodmTimer.stop()
            self.webodmTimer.deleteLater()

        self.webodmTimer = qt.QTimer()
        self.webodmTimer.setInterval(5000)
        self.webodmTimer.timeout.connect(self.checkWebODMTaskStatus)
        self.webodmTimer.start()
        
        # Enable save button if it exists (currently commented out in UI)
        if hasattr(self.widget, 'saveTaskButton') and self.widget.saveTaskButton:
            self.widget.saveTaskButton.enabled = True
    
    def generateShortTaskName(self, basePrefix, paramsDict):
        """
        Generate a short task name based on prefix and parameter hash.
        Similar to PhotoMasking's generateShortTaskName method.
        """
        import hashlib
        import json
        
        # Convert params to a stable JSON string
        paramsStr = json.dumps(paramsDict, sort_keys=True)
        hashObj = hashlib.sha256(paramsStr.encode('utf-8'))
        shortHash = hashObj.hexdigest()[:8]
        return f"{basePrefix}_{shortHash}"

    def onStopMonitoring(self):
        """
        Stop monitoring the current task (task continues on server)
        """
        if self.webodmTimer:
            self.webodmTimer.stop()
            self.webodmTimer.deleteLater()
            self.webodmTimer = None
        self.webodmTask = None
        self.widget.stopMonitoringButton.setEnabled(False)
        self.widget.webodmLogTextEdit.append("Stopped monitoring.")

    def checkWebODMTaskStatus(self):
        """
        Poll the WebODM task for status and output updates
        """
        if not self.webodmTask:
            return
        try:
            info = self.webodmTask.info(with_output=self.lastWebODMOutputLineIndex)
        except Exception as e:
            self.widget.webodmLogTextEdit.append(f"Error retrieving task info: {str(e)}")
            slicer.app.processEvents()
            return

        newLines = info.output or []
        if len(newLines) > 0:
            for line in newLines:
                self.widget.webodmLogTextEdit.append(line)
            self.lastWebODMOutputLineIndex += len(newLines)

        self.widget.webodmLogTextEdit.append(f"Status: {info.status.name}, Progress: {info.progress}%")
        cursor = self.widget.webodmLogTextEdit.textCursor()
        cursor.movePosition(qt.QTextCursor.End)
        self.widget.webodmLogTextEdit.setTextCursor(cursor)
        self.widget.webodmLogTextEdit.ensureCursorVisible()
        slicer.app.processEvents()

        if info.status.name.lower() == "completed":
            self.widget.webodmLogTextEdit.append(f"Task completed! Downloading results to {self.webodmOutDir} ...")
            slicer.app.processEvents()
            try:
                self.webodmTask.download_assets(self.webodmOutDir)
                slicer.util.infoDisplay(f"Results downloaded to:\\n{self.webodmOutDir}")
            except Exception as e:
                slicer.util.warningDisplay(f"Download failed: {str(e)}")

            if self.webodmTimer:
                self.webodmTimer.stop()
                self.webodmTimer.deleteLater()
                self.webodmTimer = None
            self.webodmTask = None
            self.widget.stopMonitoringButton.setEnabled(False)
        elif info.status.name.lower() in ["failed", "canceled"]:
            self.widget.webodmLogTextEdit.append("Task failed or canceled. Stopping.")
            slicer.app.processEvents()
            if self.webodmTimer:
                self.webodmTimer.stop()
                self.webodmTimer.deleteLater()
                self.webodmTimer = None
            self.webodmTask = None
            self.widget.stopMonitoringButton.setEnabled(False)

    def onImportModelClicked(self):
        """
        Import the reconstructed 3D model (OBJ) into Slicer
        """
        inputFolder = self.widget.inputFolderSelector.directory
        if not inputFolder or not os.path.isdir(inputFolder):
            slicer.util.errorDisplay("No input folder selected.")
            return

        # Search for WebODM output folders
        webodm_dirs = [d for d in os.listdir(inputFolder) if d.startswith("WebODM_")]
        if not webodm_dirs:
            slicer.util.errorDisplay("No WebODM output folders found.")
            return

        # For now, use the first one found (could add selection dialog)
        latest_dir = os.path.join(inputFolder, webodm_dirs[-1])
        odm_texturing = os.path.join(latest_dir, "odm_texturing")
        
        if not os.path.isdir(odm_texturing):
            slicer.util.errorDisplay(f"odm_texturing folder not found in {latest_dir}")
            return

        obj_files = [f for f in os.listdir(odm_texturing) if f.lower().endswith('.obj')]
        if not obj_files:
            slicer.util.errorDisplay("No .obj file found in odm_texturing.")
            return

        obj_path = os.path.join(odm_texturing, obj_files[0])
        slicer.util.infoDisplay(f"Importing model from:\\n{obj_path}")

        try:
            modelNode = slicer.util.loadModel(obj_path)
            if modelNode:
                slicer.util.setSliceViewerLayers(background=None, foreground=None, label=None, fit=True)
                layoutManager = slicer.app.layoutManager()
                layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
                slicer.util.infoDisplay("Model imported successfully!")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to import model:\\n{str(e)}")


class ODMLogic(ScriptedLoadableModuleLogic):
    """
    Logic class for ODM module
    Currently minimal - most logic is in ODMManager
    """
    pass

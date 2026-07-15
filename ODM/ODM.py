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
        self.cancelTaskButton = None
        self.reconnectTaskButton = None
        self.nodeDashboardLabel = None
        
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
            "matcher-type": ["flann", "bow", "bruteforce"],
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

        self.nodeDashboardLabel = qt.QLabel()
        self.nodeDashboardLabel.setOpenExternalLinks(True)
        self.nodeDashboardLabel.setToolTip(
            "Opens the NodeODM dashboard in your browser.\n"
            "Shows every task on this node (running, queued and finished) and lets you cancel them."
        )
        webodmTaskFormLayout.addRow("Node Dashboard:", self.nodeDashboardLabel)
        self.nodeIPLineEdit.connect('textChanged(QString)', self.onNodeAddressChanged)
        self.nodePortSpinBox.connect('valueChanged(int)', self.onNodeAddressChanged)
        self.updateNodeDashboardLink()

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
        
        # Max concurrency - default to number of CPUs minus 1
        cpu_count = os.cpu_count() or 16
        default_concurrency = max(1, cpu_count - 1)
        
        self.maxConcurrencySpinBox = qt.QSpinBox()
        self.maxConcurrencySpinBox.setRange(1, 256)
        self.maxConcurrencySpinBox.setValue(default_concurrency)
        self.maxConcurrencySpinBox.setToolTip(
            f"Maximum number of processes used by WebODM.\n"
            f"Default: {default_concurrency} (system CPUs - 1).\n"
            "Higher values = faster but more memory usage."
        )
        webodmTaskFormLayout.addRow("max-concurrency:", self.maxConcurrencySpinBox)
        
        # TODO: Add WebODM parameter controls here (will be added in next step)
        
        self.datasetNameLineEdit = qt.QLineEdit("SlicerReconstruction")
        self.datasetNameLineEdit.setToolTip("Name of the dataset in WebODM.\nThis will be the reconstruction folder label.")
        webodmTaskFormLayout.addRow("name:", self.datasetNameLineEdit)
        
        self.launchWebODMTaskButton = qt.QPushButton("Run NodeODM Task With Selected Parameters")
        self.launchWebODMTaskButton.setToolTip(
            "Uploads the images and queues one reconstruction task on the node.\n"
            "Uploading can take several minutes for large image sets."
        )
        webodmTaskFormLayout.addWidget(self.launchWebODMTaskButton)
        self.launchWebODMTaskButton.setEnabled(True)
        
        self.webodmLogTextEdit = qt.QTextEdit()
        self.webodmLogTextEdit.setReadOnly(True)
        webodmTaskFormLayout.addRow("Console Log:", self.webodmLogTextEdit)
        
        taskControlRow = qt.QHBoxLayout()
        self.stopMonitoringButton = qt.QPushButton("Stop Monitoring")
        self.stopMonitoringButton.setEnabled(False)
        self.stopMonitoringButton.setToolTip(
            "Stop updating this log. The task keeps running on the node.\n"
            "Use 'Cancel Task' to actually stop it."
        )
        taskControlRow.addWidget(self.stopMonitoringButton)

        self.cancelTaskButton = qt.QPushButton("Cancel Task")
        self.cancelTaskButton.setEnabled(False)
        self.cancelTaskButton.setToolTip("Cancel the monitored task on the node itself.")
        taskControlRow.addWidget(self.cancelTaskButton)
        webodmTaskFormLayout.addRow(taskControlRow)

        self.reconnectTaskButton = qt.QPushButton("Reconnect to Task on Node...")
        self.reconnectTaskButton.setToolTip(
            "Pick up a task that is already on the node - one left behind by 'Stop Monitoring',\n"
            "or started in an earlier Slicer session - and resume watching it, downloading the\n"
            "results when it finishes.\n\n"
            "NodeODM deletes finished tasks after 48 hours by default, so this only reaches\n"
            "tasks the node still remembers."
        )
        webodmTaskFormLayout.addRow(self.reconnectTaskButton)
        
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
        self.cancelTaskButton.connect('clicked(bool)', self.onCancelTaskClicked)
        self.reconnectTaskButton.connect('clicked(bool)', self.onReconnectTaskClicked)
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
            slicer.util.errorDisplay(f"Failed to create or set permissions for WebODM folder:\n{str(e)}")
    
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
            "The ODM module requires the 'pyodm' Python package.\n\n"
            "Install it now?",
            "Install pyodm"
        ):
            slicer.util.warningDisplay(
                "pyodm is required for this module to function.\n"
                "You can install it manually via:\n"
                "pip install pyodm"
            )
            return
        
        try:
            slicer.util.pip_install("pyodm")
            slicer.util.infoDisplay("pyodm installed successfully!")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to install pyodm:\n{e}\n\nPlease install manually:\npip install pyodm")

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

    def onCancelTaskClicked(self):
        """Cancel the monitored task on the node"""
        self.webODMManager.onCancelTaskClicked()

    def onReconnectTaskClicked(self):
        """Resume monitoring a task that is already on the node"""
        self.webODMManager.onReconnectTaskClicked()

    def nodeDashboardUrl(self):
        """URL of the NodeODM dashboard for the currently configured node."""
        return f"http://{self.nodeIPLineEdit.text.strip()}:{self.nodePortSpinBox.value}"

    def onNodeAddressChanged(self, unusedValue=None):
        """Keep the dashboard link in sync with the IP/port fields."""
        self.updateNodeDashboardLink()

    def updateNodeDashboardLink(self):
        url = self.nodeDashboardUrl()
        self.nodeDashboardLabel.setText(f'<a href="{url}">{url}</a>')

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
            "stop any running container on port 3002, and launch a new one.\n\n"
            "Proceed?"
        )
        if not proceed:
            slicer.util.infoDisplay("Launch NodeODM canceled by user.")
            return
        
        # Check Docker is available
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
        except Exception as e:
            slicer.util.warningDisplay(f"Docker not found or not in PATH.\nError: {str(e)}")
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
            slicer.util.errorDisplay(f"Failed to launch WebODM container:\n{str(e)}")

    def onStopNodeClicked(self):
        """
        Stop the running NodeODM container on port 3002
        """
        jobInProgress = (self.webodmTask is not None)

        if jobInProgress:
            proceed = slicer.util.confirmYesNoDisplay(
                "A WebODM task appears to be in progress. Stopping the node now will cancel that task.\n\n"
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
            slicer.util.errorDisplay(f"Failed to stop container:\n{str(e)}")

    def _updateTaskButtons(self):
        """
        Single source of truth for the task buttons: Run is available only when no task
        is tracked, and the task controls only when one is. Re-enabling Run while a task
        is still tracked is what let a second click orphan the first task.
        """
        hasTask = self.webodmTask is not None
        self.widget.launchWebODMTaskButton.setEnabled(not hasTask)
        self.widget.reconnectTaskButton.setEnabled(not hasTask)
        self.widget.stopMonitoringButton.setEnabled(hasTask)
        self.widget.cancelTaskButton.setEnabled(hasTask)

    def onRunWebODMTask(self):
        """
        Create and execute a WebODM reconstruction task.

        Run is disabled for the whole call (the upload runs on the UI thread, so clicks
        would otherwise queue up behind it) and stays disabled afterwards for as long as
        the created task is still being tracked.
        """
        self.widget.launchWebODMTaskButton.setEnabled(False)
        try:
            self._runWebODMTask()
        finally:
            self._updateTaskButtons()

    def _runWebODMTask(self):
        # Slicer tracks exactly one task. Starting another would overwrite webodmTask and
        # webodmTimer, leaving the previous one running on the node with no way to monitor
        # or cancel it from here.
        if self.webodmTask is not None:
            slicer.util.warningDisplay(
                "A task is already being monitored.\n\n"
                "Use 'Cancel Task' to stop it, or 'Stop Monitoring' to leave it running on "
                "the node, before starting another one."
            )
            return

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
            slicer.util.errorDisplay(f"Failed to connect to Node at {node_ip}:{node_port}\n{str(e)}")
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

        # Node.__init__ does not talk to the network, so this is the first call that
        # actually proves the node is up -- and it tells us what is already queued.
        dashboardUrl = self.widget.nodeDashboardUrl()
        try:
            nodeInfo = node.info()
        except Exception as e:
            slicer.util.errorDisplay(
                f"Could not reach NodeODM at {dashboardUrl}\n\n{str(e)}\n\n"
                "Is the node running? Use 'Launch NodeODM' above to start it."
            )
            return

        queuedCount = nodeInfo.task_queue_count or 0
        if queuedCount > 0:
            proceed = slicer.util.confirmYesNoDisplay(
                f"This node already has {queuedCount} task(s) running or queued.\n\n"
                "A new task waits in line behind them, and will just report "
                "'Queued, Progress: 0%' until they finish.\n\n"
                f"Review or cancel existing tasks at:\n{dashboardUrl}\n\n"
                "Queue another task anyway?"
            )
            if not proceed:
                return

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

        # create_task uploads every file on the UI thread, which for a few hundred images
        # means minutes of frozen application. Drive a progress dialog from pyodm's
        # callback so the upload is visibly alive.
        fileCount = len(files_to_upload)
        progressDialog = slicer.util.createProgressDialog(
            windowTitle="Creating NodeODM Task",
            labelText=f"Uploading {fileCount} file(s) to NodeODM...",
            maximum=100
        )
        progressDialog.setCancelButton(None)
        progressDialog.show()
        slicer.app.processEvents()

        def onUploadProgress(percent):
            progressDialog.labelText = f"Uploading {fileCount} file(s) to NodeODM... {percent:.0f}%"
            progressDialog.setValue(int(percent))
            slicer.app.processEvents()

        # Report the error only after the progress dialog is down: errorDisplay is modal,
        # so raising it from the except block would strand the upload dialog behind it.
        createError = None
        try:
            self.webodmTask = node.create_task(
                files=files_to_upload,
                options=params,
                name=shortTaskName,
                progress_callback=onUploadProgress
            )
        except Exception as e:
            createError = e
        finally:
            progressDialog.close()
            slicer.app.processEvents()

        if createError is not None:
            slicer.util.errorDisplay(f"Task creation failed:\n{str(createError)}")
            return

        slicer.util.infoDisplay(
            f"Task '{shortTaskName}' created successfully. Monitoring progress...",
            autoCloseMsec=3000
        )

        # Path only -- the folder is created at download time. Creating it here left an
        # empty WebODM_<name> folder behind whenever a task was never downloaded (Stop
        # Monitoring, a failure, a Slicer restart), which then looked like output and
        # sent 'Import Reconstructed Model' hunting for an odm_texturing that never came.
        self.webodmOutDir = os.path.join(inputFolder, f"WebODM_{shortTaskName}")

        self.widget.webodmLogTextEdit.clear()
        self.widget.webodmLogTextEdit.append(f"Task '{shortTaskName}' queued on {dashboardUrl}")
        self.widget.webodmLogTextEdit.append(f"  node task UUID: {self.webodmTask.uuid}")
        self.widget.webodmLogTextEdit.append(f"  results will download to: {self.webodmOutDir}")

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
        Name a task uniquely: prefix, a hash of the parameters, and the start time.

        The hash alone is a pure function of the parameters, so re-running the same
        settings produced the same name every time -- two tasks on the node sharing one
        name, both pointing at one WebODM_<name> folder, the second download landing on
        top of the first. The timestamp is what makes a run identifiable; the hash is
        kept so the settings are still readable at a glance.
        """
        import hashlib
        import json
        import datetime

        # Convert params to a stable JSON string
        paramsStr = json.dumps(paramsDict, sort_keys=True)
        hashObj = hashlib.sha256(paramsStr.encode('utf-8'))
        shortHash = hashObj.hexdigest()[:8]
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{basePrefix}_{shortHash}_{stamp}"

    def onStopMonitoring(self):
        """
        Stop monitoring the current task (task continues on server)
        """
        self._stopMonitoringTimer()
        self.webodmTask = None
        self._updateTaskButtons()
        self.widget.webodmLogTextEdit.append(
            "Stopped monitoring. The task is still running on the node -- see "
            f"{self.widget.nodeDashboardUrl()} to follow or cancel it."
        )

    def _stopMonitoringTimer(self):
        if self.webodmTimer:
            self.webodmTimer.stop()
            self.webodmTimer.deleteLater()
            self.webodmTimer = None

    def onCancelTaskClicked(self):
        """
        Cancel the monitored task on the node itself (unlike Stop Monitoring,
        which only stops Slicer from watching it).
        """
        if not self.webodmTask:
            slicer.util.infoDisplay("No task is currently being monitored.")
            return

        if not slicer.util.confirmYesNoDisplay(
            "Cancel the monitored task on the node?\n\nThis cannot be undone."
        ):
            return

        try:
            self.webodmTask.cancel()
        except Exception as e:
            slicer.util.warningDisplay(
                f"Failed to cancel the task:\n{str(e)}\n\n"
                f"You can also cancel it at {self.widget.nodeDashboardUrl()}"
            )
            return

        self.widget.webodmLogTextEdit.append("Task canceled on the node.")
        self._stopMonitoringTimer()
        self.webodmTask = None
        self._updateTaskButtons()

    def describeTask(self, info):
        """One line per task for the reconnect picker."""
        stamp = info.date_created.strftime("%Y-%m-%d %H:%M") if info.date_created else "?"
        return (f"{info.name or '(unnamed)'}  |  {info.status.name}  |  "
                f"{stamp} UTC  |  {info.images_count} images")

    def onReconnectTaskClicked(self):
        """
        Attach to a task that is already on the node -- left running by Stop Monitoring,
        or started in an earlier Slicer session -- and resume monitoring it, so its
        results can still be downloaded.
        """
        if self.webodmTask is not None:
            slicer.util.warningDisplay(
                "A task is already being monitored.\n\nUse 'Stop Monitoring' first."
            )
            return

        inputFolder = self.widget.inputFolderSelector.directory
        if not inputFolder or not os.path.isdir(inputFolder):
            slicer.util.errorDisplay(
                "Select the masked images folder first - it is where the results are downloaded to."
            )
            return

        try:
            from pyodm import Node
        except ImportError:
            slicer.util.errorDisplay("pyodm module not found. Please install it via pip.")
            return

        dashboardUrl = self.widget.nodeDashboardUrl()
        node = Node(self.widget.nodeIPLineEdit.text.strip(), self.widget.nodePortSpinBox.value)

        try:
            entries = node.get('/task/list') or []
        except Exception as e:
            slicer.util.errorDisplay(
                f"Could not reach NodeODM at {dashboardUrl}\n\n{str(e)}\n\n"
                "Is the node running? Use 'Launch NodeODM' above to start it."
            )
            return

        if not entries:
            slicer.util.infoDisplay(f"The node at {dashboardUrl} has no tasks.")
            return

        # /task/list returns UUIDs only, so ask each task for its name/status/date.
        tasks = []
        for entry in entries:
            uuid = entry.get('uuid') if isinstance(entry, dict) else None
            if not uuid:
                continue
            try:
                tasks.append(node.get_task(uuid).info())
            except Exception:
                continue  # vanished between listing and asking; skip it

        if not tasks:
            slicer.util.warningDisplay(
                f"The node at {dashboardUrl} listed tasks, but none of them could be read."
            )
            return

        tasks.sort(key=lambda i: i.date_created, reverse=True)
        labels = [self.describeTask(i) for i in tasks]

        # PythonQt's binding for QInputDialog.getItem returns just the selected string
        # (empty on cancel), not the (value, ok) tuple the C++/PyQt API uses.
        chosen = qt.QInputDialog.getItem(
            slicer.util.mainWindow(),
            "Reconnect to Task",
            f"Tasks on {dashboardUrl}:",
            labels,
            0,
            False,
        )
        if not chosen or chosen not in labels:
            return

        self.attachToTask(node, tasks[labels.index(chosen)], inputFolder, dashboardUrl)

    def attachToTask(self, node, info, inputFolder, dashboardUrl):
        """Resume monitoring an existing node task and route its results to disk."""
        self.webodmTask = node.get_task(info.uuid)

        # The task name is the folder key (WebODM_<name>), which is why names carry a
        # timestamp. A task created outside Slicer may be unnamed; fall back to its UUID.
        folderKey = info.name or f"task_{info.uuid[:8]}"
        self.webodmOutDir = os.path.join(inputFolder, f"WebODM_{folderKey}")

        self.widget.webodmLogTextEdit.clear()
        self.widget.webodmLogTextEdit.append(f"Reconnected to '{folderKey}' on {dashboardUrl}")
        self.widget.webodmLogTextEdit.append(f"  node task UUID: {info.uuid}")
        self.widget.webodmLogTextEdit.append(f"  results will download to: {self.webodmOutDir}")

        self.lastWebODMOutputLineIndex = 0
        self._stopMonitoringTimer()
        self.webodmTimer = qt.QTimer()
        self.webodmTimer.setInterval(5000)
        self.webodmTimer.timeout.connect(self.checkWebODMTaskStatus)
        self.webodmTimer.start()
        self._updateTaskButtons()

        # Poll once now rather than making the user wait for the first tick -- and if the
        # task already finished, this downloads it immediately.
        self.checkWebODMTaskStatus()

    def queuedTaskCount(self):
        """
        How many tasks are QUEUED or RUNNING on the node (this one included), or None if
        it can't be asked. Used to explain a task sitting at 'Queued, Progress: 0%'.
        """
        try:
            from pyodm import Node
            node = Node(self.widget.nodeIPLineEdit.text.strip(), self.widget.nodePortSpinBox.value)
            return node.info().task_queue_count
        except Exception:
            return None

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

        statusLine = f"Status: {info.status.name}, Progress: {info.progress}%"
        if info.status.name.lower() == "queued":
            # "Queued, 0%" on its own reads as a hang. Say what it is waiting behind.
            # NodeODM counts QUEUED+RUNNING including this task, so drop it to get the
            # number actually ahead.
            totalCount = self.queuedTaskCount()
            if totalCount is not None and totalCount > 1:
                statusLine += (
                    f" -- {totalCount - 1} task(s) ahead of yours on this node. "
                    f"See {self.widget.nodeDashboardUrl()}"
                )
        self.widget.webodmLogTextEdit.append(statusLine)
        cursor = self.widget.webodmLogTextEdit.textCursor()
        cursor.movePosition(qt.QTextCursor.End)
        self.widget.webodmLogTextEdit.setTextCursor(cursor)
        self.widget.webodmLogTextEdit.ensureCursorVisible()
        slicer.app.processEvents()

        if info.status.name.lower() == "completed":
            self.widget.webodmLogTextEdit.append(f"Task completed! Downloading results to {self.webodmOutDir} ...")
            slicer.app.processEvents()
            try:
                os.makedirs(self.webodmOutDir, exist_ok=True)
                self.webodmTask.download_assets(self.webodmOutDir)
                slicer.util.infoDisplay(f"Results downloaded to:\n{self.webodmOutDir}")
            except Exception as e:
                slicer.util.warningDisplay(f"Download failed: {str(e)}")

            self._stopMonitoringTimer()
            self.webodmTask = None
            self._updateTaskButtons()
        elif info.status.name.lower() in ["failed", "canceled"]:
            self.widget.webodmLogTextEdit.append("Task failed or canceled. Stopping.")
            slicer.app.processEvents()
            self._stopMonitoringTimer()
            self.webodmTask = None
            self._updateTaskButtons()

    def onImportModelClicked(self):
        """
        Import the reconstructed 3D model (OBJ) into Slicer
        """
        inputFolder = self.widget.inputFolderSelector.directory
        if not inputFolder or not os.path.isdir(inputFolder):
            slicer.util.errorDisplay("No input folder selected.")
            return

        # Search for WebODM output folders. os.listdir is in filesystem order, so sort:
        # task names end in a YYYYMMDD-HHMMSS stamp, which sorts chronologically, making
        # the last entry genuinely the newest rather than whichever the OS happened to
        # hand back last.
        webodm_dirs = sorted(d for d in os.listdir(inputFolder) if d.startswith("WebODM_"))
        if not webodm_dirs:
            slicer.util.errorDisplay("No WebODM output folders found.")
            return

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
        slicer.util.infoDisplay(f"Importing model from:\n{obj_path}")

        try:
            modelNode = slicer.util.loadModel(obj_path)
            if modelNode:
                slicer.util.setSliceViewerLayers(background=None, foreground=None, label=None, fit=True)
                layoutManager = slicer.app.layoutManager()
                layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
                slicer.util.infoDisplay("Model imported successfully!")
        except Exception as e:
            slicer.util.errorDisplay(f"Failed to import model:\n{str(e)}")


class ODMLogic(ScriptedLoadableModuleLogic):
    """
    Logic class for ODM module
    Currently minimal - most logic is in ODMManager
    """
    pass

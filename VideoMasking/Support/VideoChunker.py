# -*- coding: utf-8 -*-
"""
VideoChunker - Support library for splitting large videos into manageable chunks
--------------------------------------------------------------------------------
Handles video chunking logic to prevent OOM when processing long videos with SAM.

Strategy:
- Videos ≤600 frames: Process as single chunk (no splitting)
- Videos >600 frames: Split into 360-600 frame chunks using ffmpeg segment muxer
- Maximum supported: 2000 frames total (~33s @ 60fps)

Memory benefits:
- Single chunk in RAM at a time (max ~8GB for 4K)
- VRAM per chunk: ~15GB vs 47GB+ for full video
- Sequential processing with cleanup between chunks
"""

import os
import subprocess
import shlex
import shutil
from pathlib import Path


class VideoChunker:
    """Manages video chunking for memory-efficient processing."""
    
    # Frame limits
    MAX_FRAMES_TOTAL = 2000  # Hard limit (~33s @ 60fps)
    MIN_CHUNK_FRAMES = 240   # 4s @ 60fps (safe for 4K on 48GB GPU)
    MAX_CHUNK_FRAMES = 400   # ~6.7s @ 60fps (~13.3GB VRAM for 4K)
    
    def __init__(self, video_path=None, output_dir=None, logger=None):
        """
        Initialize chunker.
        
        Args:
            video_path: Path to source video (MP4)
            output_dir: Directory for chunks and frames
            logger: Callable for logging (e.g., self._log from VideoMasking)
        """
        self.video_path = Path(video_path) if video_path else None
        self.output_dir = Path(output_dir) if output_dir else None
        self.logger = logger or print
        self._metrics = None
    
    def _log(self, msg):
        """Internal logging wrapper."""
        try:
            self.logger(msg)
        except Exception:
            print(msg)
    
    def probe_video_metrics(self, video_path=None):
        """
        Fast probe of video metrics without decoding frames.
        
        Args:
            video_path: Optional override of instance video_path
            
        Returns:
            dict with keys: width, height, fps, frames (ints/floats), or {} on failure
        """
        import cv2
        
        vpath = video_path or self.video_path
        if not vpath:
            return {}
        
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            return {}
        
        try:
            metrics = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
            self._metrics = metrics
            return metrics
        except Exception as e:
            self._log(f"WARNING: Video probe failed: {e}")
            return {}
        finally:
            cap.release()
    
    def validate_frame_count(self):
        """
        Check if video exceeds maximum frame limit.
        
        Returns:
            tuple: (is_valid: bool, message: str)
        """
        if not self._metrics:
            self._metrics = self.probe_video_metrics()
        
        total_frames = self._metrics.get("frames", 0)
        
        if total_frames == 0:
            return False, "Could not determine video frame count"
        
        if total_frames > self.MAX_FRAMES_TOTAL:
            duration_s = total_frames / self._metrics.get("fps", 60.0)
            max_duration_s = self.MAX_FRAMES_TOTAL / self._metrics.get("fps", 60.0)
            return False, (
                f"Video has {total_frames} frames ({duration_s:.1f}s) - exceeds maximum of "
                f"{self.MAX_FRAMES_TOTAL} frames ({max_duration_s:.1f}s).\n\n"
                f"Please trim the video to ≤{self.MAX_FRAMES_TOTAL} frames and try again."
            )
        
        return True, f"Video OK: {total_frames} frames"
    
    def needs_chunking(self):
        """
        Determine if video requires chunking.
        
        Returns:
            bool: True if video has more than MAX_CHUNK_FRAMES
        """
        if not self._metrics:
            self._metrics = self.probe_video_metrics()
        
        total_frames = self._metrics.get("frames", 0)
        return total_frames > self.MAX_CHUNK_FRAMES
    
    def create_chunks(self):
        """
        Split video into chunks using ffmpeg segment muxer.
        
        Returns:
            list[dict]: Chunk metadata with keys:
                - video_path: Path to chunk MP4
                - start_frame: Global start frame index
                - end_frame: Global end frame index (inclusive)
                - num_frames: Number of frames in chunk
        """
        if not self.video_path or not self.video_path.exists():
            raise RuntimeError(f"Video path not set or doesn't exist: {self.video_path}")
        
        if not self.output_dir:
            raise RuntimeError("Output directory not set")
        
        if not self._metrics:
            self._metrics = self.probe_video_metrics()
        
        total_frames = self._metrics.get("frames", 0)
        fps = self._metrics.get("fps", 60.0)
        
        if total_frames == 0:
            raise RuntimeError("Could not determine video frame count")
        
        # Calculate optimal chunk size (aim for middle of range)
        num_chunks = max(1, (total_frames + self.MAX_CHUNK_FRAMES - 1) // self.MAX_CHUNK_FRAMES)
        chunk_size = min(self.MAX_CHUNK_FRAMES, max(self.MIN_CHUNK_FRAMES, total_frames // num_chunks))
        
        # Create chunks directory
        chunks_dir = self.output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate segment duration in seconds
        segment_duration = chunk_size / fps
        
        # Run ffmpeg segmenter (stream copy - no re-encode)
        segment_pattern = str(chunks_dir / "chunk_%03d.mp4")
        ffmpeg_cmd = (
            f'ffmpeg -y -i "{self.video_path}" '
            f'-c copy -f segment -segment_time {segment_duration} '
            f'-reset_timestamps 1 "{segment_pattern}"'
        )
        
        self._log(f"Splitting video into chunks ({chunk_size} frames/chunk, ~{segment_duration:.1f}s each)...")
        
        try:
            # Use clean environment to avoid library conflicts
            env = os.environ.copy()
            # Remove potentially conflicting library paths
            env.pop('LD_LIBRARY_PATH', None)
            
            result = subprocess.run(
                shlex.split(ffmpeg_cmd),
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            if result.stderr:
                # ffmpeg outputs to stderr even on success
                for line in result.stderr.split('\n'):
                    if 'error' in line.lower() or 'failed' in line.lower():
                        self._log(f"ffmpeg: {line}")
        except subprocess.CalledProcessError as e:
            self._log(f"ffmpeg error output:\n{e.stderr}")
            raise RuntimeError(f"Video splitting failed: {e}")
        
        # Build metadata for each chunk
        chunk_files = sorted(chunks_dir.glob("chunk_*.mp4"))
        if not chunk_files:
            raise RuntimeError(f"No chunk files created in {chunks_dir}")
        
        metadata = []
        current_frame = 0
        
        for idx, chunk_path in enumerate(chunk_files):
            # Probe actual frame count (last chunk may be shorter)
            chunk_metrics = self.probe_video_metrics(str(chunk_path))
            chunk_frames = chunk_metrics.get("frames", chunk_size)
            
            metadata.append({
                "video_path": str(chunk_path),
                "start_frame": current_frame,
                "end_frame": current_frame + chunk_frames - 1,
                "num_frames": chunk_frames
            })
            current_frame += chunk_frames
        
        self._log(f"Created {len(metadata)} chunks (total {current_frame} frames)")
        return metadata
    
    def extract_first_frame_only(self, chunk_metadata):
        """
        Extract only the first frame from the first chunk for ROI setup.
        
        Args:
            chunk_metadata: List of chunk metadata dicts (from create_chunks)
            
        Returns:
            Path: Directory containing the single first frame
        """
        if not chunk_metadata:
            raise RuntimeError("No chunk metadata provided")
        
        first_chunk_path = chunk_metadata[0]["video_path"]
        first_frame_dir = self.output_dir / "first_frame_only"
        first_frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frame 1 as 000001.jpg (to match frame numbering convention)
        output_path = first_frame_dir / "000001.jpg"
        ffmpeg_cmd = f'ffmpeg -y -i "{first_chunk_path}" -vframes 1 -q:v 2 "{output_path}"'
        
        self._log(f"Extracting first frame for ROI setup...")
        
        try:
            # Use clean environment to avoid library conflicts
            env = os.environ.copy()
            env.pop('LD_LIBRARY_PATH', None)
            
            subprocess.run(
                shlex.split(ffmpeg_cmd),
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"First frame extraction failed: {e.stderr}")
        
        if not output_path.exists():
            raise RuntimeError(f"First frame not created: {output_path}")
        
        self._log(f"First frame ready: {output_path}")
        return first_frame_dir
    
    def extract_chunk_frames(self, chunk_info, output_dir):
        """
        Extract frames for a specific chunk on-demand.
        
        Args:
            chunk_info: Single chunk metadata dict
            output_dir: Base output directory
            
        Returns:
            Path: Directory containing extracted frames for this chunk
        """
        chunk_path = chunk_info["video_path"]
        chunk_idx = int(Path(chunk_path).stem.split('_')[-1])  # Extract number from chunk_XXX.mp4
        
        chunk_frames_dir = Path(output_dir) / f"chunk_{chunk_idx:03d}_frames"
        chunk_frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames with sequential numbering starting from 1
        output_pattern = chunk_frames_dir / "%06d.jpg"
        ffmpeg_cmd = f'ffmpeg -y -i "{chunk_path}" -q:v 2 "{output_pattern}"'
        
        self._log(f"Extracting frames for chunk {chunk_idx}...")
        
        try:
            # Use clean environment to avoid library conflicts
            env = os.environ.copy()
            env.pop('LD_LIBRARY_PATH', None)
            
            subprocess.run(
                shlex.split(ffmpeg_cmd),
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Chunk frame extraction failed: {e.stderr}")
        
        # Verify frames were created
        frame_files = list(chunk_frames_dir.glob("*.jpg"))
        if not frame_files:
            raise RuntimeError(f"No frames extracted to {chunk_frames_dir}")
        
        self._log(f"Extracted {len(frame_files)} frames for chunk {chunk_idx}")
        return chunk_frames_dir
    
    def cleanup_chunk_frames(self, chunk_frames_dir):
        """
        Remove chunk frame directory to free disk space.
        
        Args:
            chunk_frames_dir: Path to chunk frames directory
        """
        if not chunk_frames_dir:
            return
        
        chunk_path = Path(chunk_frames_dir)
        if chunk_path.exists() and chunk_path.is_dir():
            try:
                shutil.rmtree(chunk_path)
                self._log(f"Cleaned up chunk frames: {chunk_path.name}")
            except Exception as e:
                self._log(f"WARNING: Could not clean up {chunk_path}: {e}")
    
    def cleanup_all_chunks(self):
        """
        Remove all chunk-related directories and files.
        Useful for final cleanup after processing completes.
        """
        if not self.output_dir:
            return
        
        # Clean up chunks directory
        chunks_dir = self.output_dir / "chunks"
        if chunks_dir.exists():
            try:
                shutil.rmtree(chunks_dir)
                self._log(f"Removed chunks directory: {chunks_dir}")
            except Exception as e:
                self._log(f"WARNING: Could not remove chunks directory: {e}")
        
        # Clean up first frame directory
        first_frame_dir = self.output_dir / "first_frame_only"
        if first_frame_dir.exists():
            try:
                shutil.rmtree(first_frame_dir)
                self._log(f"Removed first frame directory")
            except Exception as e:
                self._log(f"WARNING: Could not remove first frame directory: {e}")
        
        # Clean up any lingering chunk frame directories
        for chunk_dir in self.output_dir.glob("chunk_*_frames"):
            try:
                shutil.rmtree(chunk_dir)
                self._log(f"Removed lingering chunk frames: {chunk_dir.name}")
            except Exception as e:
                self._log(f"WARNING: Could not remove {chunk_dir}: {e}")

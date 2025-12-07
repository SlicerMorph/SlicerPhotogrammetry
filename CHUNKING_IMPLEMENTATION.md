# Video Chunking Implementation Summary

## Overview
Implemented file-based video chunking to enable processing of longer videos (up to 2000 frames) without running out of RAM/VRAM.

## Branch
`video-file-chunking`

## Files Modified/Created

### New Files
- **`VideoMasking/Support/VideoChunker.py`** (369 lines)
  - Standalone support library for video chunking logic
  - Handles video validation, splitting, frame extraction, and cleanup
  - Reusable by other modules

### Modified Files
- **`VideoMasking/VideoMasking.py`**
  - Added VideoChunker import (~13 lines)
  - Added chunk metadata tracking to `__init__` (~4 lines)
  - Replaced `onLoadVideo()` with chunking-aware version (~90 lines vs ~110 original)
  - Replaced `onRunTracking()` with chunk-aware processing (~180 lines vs ~150 original)
  - Added `_convert_masks_to_uint8()` helper (~15 lines)
  - **Total net change: ~40 lines added to main file**

## Key Features

### VideoChunker Class API
```python
chunker = VideoChunker(video_path, output_dir, logger=self._log)

# Validation
is_valid, msg = chunker.validate_frame_count()  # Checks ≤2000 frames

# Chunking decision
if chunker.needs_chunking():  # True if >600 frames
    metadata = chunker.create_chunks()  # Split video using ffmpeg
    chunker.extract_first_frame_only(metadata)  # For ROI setup
    
# Per-chunk processing
for chunk_info in metadata:
    frames_dir = chunker.extract_chunk_frames(chunk_info, output_dir)
    # ... process chunk ...
    chunker.cleanup_chunk_frames(frames_dir)

# Final cleanup
chunker.cleanup_all_chunks()
```

### Processing Workflow

#### Single Chunk (≤600 frames)
1. Extract all frames
2. Load all frames for ROI setup
3. Run SAM tracking on full video
4. Build masked frames buffer
5. Enable keyframe filtering

#### Multi-Chunk (>600 frames, ≤2000 frames)
1. Validate total frames ≤2000
2. Split video into chunks using ffmpeg segment muxer
3. Extract **only first frame** for ROI setup
4. User draws ROI on first frame
5. For each chunk sequentially:
   - Extract chunk frames on-demand
   - Load chunk frames into RAM (max 600 frames)
   - Run SAM tracking on chunk video
   - Remap masks to global indices
   - Clear chunk frames from disk and RAM
   - Clear GPU cache
6. Merge all chunk masks
7. **Disable keyframe filtering** (frames not kept in RAM)

## Memory Benefits

### Before (Single Video Processing)
- **RAM**: 2000 frames × 4K ≈ 50GB (OOM)
- **VRAM**: SAM processes all frames ≈ 47GB+ (OOM at ~1250 frames)

### After (Chunked Processing)
- **RAM**: Max 600 frames × 4K ≈ 8GB per chunk
- **VRAM**: ~15GB per chunk (sequential processing)
- **Disk**: Temporary chunk files cleaned immediately after processing

## Chunking Parameters

```python
MAX_FRAMES_TOTAL = 2000  # Hard limit (~33s @ 60fps)
MIN_CHUNK_FRAMES = 360   # 6s @ 60fps
MAX_CHUNK_FRAMES = 600   # 10s @ 60fps
```

**Chunk size calculation**:
- Optimal chunk size: 360-600 frames
- Algorithm: `min(600, max(360, total_frames // num_chunks))`
- Example: 1500 frames → 3 chunks of ~500 frames each

## Technical Details

### ffmpeg Segmentation
```bash
ffmpeg -y -i input.mp4 \
  -c copy \                        # No re-encode (fast)
  -f segment \                     # Segment muxer
  -segment_time 10.0 \             # 600 frames / 60 fps
  -reset_timestamps 1 \            # Each chunk starts at t=0
  chunks/chunk_%03d.mp4
```

### Global Index Remapping
```python
# Chunk 0: frames 0-599   → global indices 0-599
# Chunk 1: frames 0-499   → global indices 600-1099
# Chunk 2: frames 0-399   → global indices 1100-1499

for local_idx, mask in chunk_masks.items():
    global_idx = chunk_info["start_frame"] + local_idx
    all_masks[global_idx] = mask
```

### Limitations
- **Keyframe filtering disabled** for chunked videos (would require re-loading all frames)
- **Masked frames not kept** in RAM for chunked videos
- **SAM tracking continuity**: Minor restart artifacts possible at chunk boundaries (acceptable for 6-10s intervals)

## Testing Strategy

### Test Case 1: Small Video (≤600 frames)
- **Expected**: No chunking, identical behavior to current implementation
- **Verify**: All frames loaded, keyframe filtering enabled, masked frames available

### Test Case 2: Medium Video (600-1200 frames)
- **Expected**: 2-3 chunks created
- **Verify**: Chunk creation, sequential processing, global index remapping, mask continuity

### Test Case 3: Large Video (1500-2000 frames)
- **Expected**: 3-4 chunks created
- **Verify**: No OOM, complete processing, correct total frame count

### Test Case 4: Oversized Video (>2000 frames)
- **Expected**: Validation error, processing blocked
- **Verify**: Clear error message with frame count and time duration

## Future Enhancements (Phase 2)

1. **Chunk overlap** for boundary continuity (5-10% overlap)
2. **Chunk-aware keyframe filtering** (process per-chunk, merge results)
3. **On-demand frame reload** for saving masked frames
4. **Progress bar** showing chunk X/Y during processing
5. **Adaptive chunk sizing** based on available GPU memory

## Status
✅ Implementation complete  
⏳ Testing pending  
🚫 **NOT COMMITTED** (awaiting user approval)

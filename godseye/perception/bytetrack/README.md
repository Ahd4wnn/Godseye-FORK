# Vendored ByteTrack tracker
Upstream: https://github.com/ifzhang/ByteTrack @ d1bf0191adff59bc8fcfeaa0b33d3d1642552a99 (MIT, see LICENSE.ByteTrack).
Files: `yolox/tracker/{basetrack,byte_tracker,kalman_filter,matching}.py`. The tracking algorithm is unchanged.
Patches (marked `[godseye patch N]` in code):
1. Track ids come from a per-tracker `id_allocator` instead of the process-global `BaseTrack._count`, so ids are camera-local and deterministic.
2. `STrack.payload` carries a detection's metadata (embedding, class, detection index) through association.
3. The hard-coded low-score threshold `0.1` is now `args.low_thresh`.
4. Compatibility: `np.float` (removed in numpy>=1.24) -> `float`; absolute `yolox.*` imports -> relative; unused torch imports removed.
Dependencies kept as upstream: `lap`, `cython_bbox`, `scipy`.

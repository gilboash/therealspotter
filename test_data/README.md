# Test data for the live-perf branch (temporary)

A learned track, ready for Live: no learning needed. Only on the `live-perf` branch; it goes when the branch does.

- `fril-27-first-track/` — the track's learned gates, gate order and timing, and map, for both models (v2:
  `gate_memory.json`, `gate_timing.learned.json`; v3: `….v3.json`), with the gate pictures (`runs/`, `runs_v3/`).
- The combo video `combo_ft35_Brackets_R4-2.mp4` (811 MB, over GitHub's 100 MB file limit) comes over Drive: put it
  in `fril-27-first-track/combo_videos/`. Its SHA-256: `2baadcb047e9b0d20c3666d6b2df143137dc9dbbfbeeef89220d2cc3a1f51328`.

Use it as the data folder, from the repo's folder:

    echo '{"data_root": "'$PWD'/test_data"}' > local_config.json

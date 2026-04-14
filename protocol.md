## INPUT
All files within the current workspace directory.

## TARGET
1. Resolve the `TypeError` in `pipeline.py` where `extract_audio()` is called without the required `cfg` positional argument.
2. Update the call site in `run_pipeline` (approximately line 117) to pass the necessary configuration object to `extract_audio`.
3. Verify that executing `pipeline.py` completes the audio extraction phase without triggering the `TypeError: extract_audio() missing 1 required positional argument: 'cfg'`.

## RESTRICTIONS
No restrictions.
# Obtain and audit SoccerNet-MVFoul

No videos, annotations, password or dataset license acceptance are bundled.
The loader expects the official distribution, not a made-up manifest.

1. Visit the [official repository](https://github.com/SoccerNet/sn-mvfoul) and
   follow its **NDA** link to request access credentials from SoccerNet.
2. On a machine where you can store the dataset, install `SoccerNet`:

   ```bash
   python -m pip install SoccerNet
   python -m scripts.download_dataset --destination /absolute/path/to/SoccerNet
   ```

   The helper prompts privately for the authorized password; it does not store it.
   Download includes Train/Valid/Test by default. Add `--challenge` if needed.
3. Extract archives locally while keeping the official naming. The expected
   structure, established from official `data_loader.py`, is:

   ```text
   SoccerNet/
     Train/
       annotations.json
       action_0/clip_0.mp4
       action_0/clip_1.mp4
       ...
     Valid/...
     Test/...
     Chall/...                 # optional, targets withheld
   ```

4. Annotations contain an `Actions` object keyed by numeric incident IDs. Fields
   include `Action class`, `Offence`, `Severity`, and `Clips`. The videos
   `clip_0`, `clip_1`, ... under the same `action_ID` are corresponding camera
   views of one incident. We use annotation keys directly, not folder count or
   JSON order, and require all annotated views to exist. Unknown schemas fail.
5. Inspect actual data before training:

   ```bash
   python -m scripts.inspect_dataset --data-root /absolute/path/to/SoccerNet
   python -m scripts.preview_incident --data-root /absolute/path/to/SoccerNet --split Valid --index 0 --output incident_preview.png
   ```

The audit reports raw/retained counts, all exclusions, eight/four class
distributions, view-count histogram and sample view paths. Use
`--annotations-only` to audit labels before videos finish downloading. The
contact sheet shows each camera on its own row; use it to verify the default
3-second center, roughly one-second window and cross-camera synchronization.

The current official README lists 2,916/411/301/273 raw actions for
Train/Valid/Test/Chall. These are upstream reference counts, not locally observed
counts. The official filtering policy removes ambiguous labels, so effective
training sizes differ. See `docs/BASELINE.md` for the 2,319-versus-2,916 caveat.

Training/evaluation refuse missing data and corrupt videos. Challenge inference
is supported when its annotation file provides `Actions`/`Clips`; absent targets
produce predictions but no metrics. If a downloaded release differs, inspect and
update the adapter explicitly, rather than invent paths or labels.

For spotting later, curated MVFoul incidents alone are insufficient to estimate
background false-alert rates. Continuous matches, event timestamps and a suitable
negative/background evaluation set will be needed.

# GUI Demo Fixture

Run the GUI against this self-contained fixture:

```bash
python gui.py --config fixtures/gui_demo/config.demo.json --port 8766
```

It points the GUI at `fixtures/gui_demo/reports` and `fixtures/gui_demo/memory` so Reports, audio playback, and Memory tabs have visible data without committing local runtime output.

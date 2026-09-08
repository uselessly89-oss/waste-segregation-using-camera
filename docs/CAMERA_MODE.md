# Camera Mode Guide

This is the easiest way to demo waste segregation — no conveyor belt or actuator required.

## Quick Start

```bash
pip install -r requirements-app.txt
python app.py
```

Open the displayed browser URL and select **Webcam**.

## Phone Camera

1. Run `python app.py` on a laptop connected to the same Wi-Fi as the phone.
2. Find the laptop's LAN IP (e.g. `192.168.1.20`).
3. Open `http://<LAPTOP-IP>:7860` in the phone browser.
4. Use the phone's rear camera to point at waste.

If port 7860 is blocked, check your firewall or use the laptop camera directly.

## Command-line Detection

For a terminal-based demo without the Gradio UI:

```bash
python src/detect.py --model models/best.pt --source 0 --show
```

- `--source 0` — default webcam
- `--source video.mp4` — a recorded video
- `--conf 0.40` — lower confidence threshold
- `--iou 0.55` — adjust NMS

## Cluster Mode (App)

When **Cluster mode** is enabled (default), the app reports every confidently
visible object and routes uncertain/hidden material to REJECT. The model does
**not** guess about items it cannot see.

For a whole mixed pile, the correct production design is segmentation plus
auxiliary sensing (depth, weight, NIR) and a second pass after items are separated.

## Important

- The app needs trained weights at `models/best.pt` (set `WASTE_MODEL` env var to use a different path).
- This camera app is for perception/demo only. Physical actuator control requires additional hardware validation.

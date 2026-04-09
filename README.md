# Robot Camera + Chat Web Interface

This project now includes a Flask web app that shows a live camera feed and a chat interface for robot commands.

## What it does

- Streams camera frames in-browser from OpenCV (`/api/video_feed`)
- Accepts chat commands and maps them to robot actions (`/api/chat`)
- Supports two chat modes:
  - `Local Parser` (no API key needed)
  - `AI Interpreter` (requires `OPENAI_API_KEY`)
- Falls back to simulation mode if Dynamixel SDK or hardware is not available

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python app.py
```

Open: `http://127.0.0.1:5000`

## Optional environment variables

```powershell
$env:CAMERA_INDEX="0"
$env:CAMERA_FALLBACK_INDICES="0,1,2"
$env:OPENAI_API_KEY="your_key_here"
$env:OPENAI_MODEL="gpt-4.1-mini"
$env:PORT="5000"
```

## Camera troubleshooting

If the camera panel is offline:

1. Close other apps using the camera (Zoom/Teams/Camera app).
2. Use the UI controls in the header to set or toggle Camera ID live.
3. Try a different index manually:

```powershell
$env:CAMERA_INDEX="1"
python app.py
```

4. Try a wider fallback list:

```powershell
$env:CAMERA_FALLBACK_INDICES="0,1,2,3"
python app.py
```

5. Check health endpoint for diagnostics:

`http://127.0.0.1:5000/api/health`

Runtime camera control endpoints:

- `POST /api/camera/set_index` with JSON body `{ "camera_index": 1 }`
- `POST /api/camera/toggle` with JSON body `{ "direction": "next" }`

## Example commands

- `connect`
- `read positions`
- `move joint 11 to 2300`
- `move base to 2400`
- `open gripper`
- `close gripper`
- `home`
- `disconnect`

## Robot port defaults

The current robot adapter defaults to:

- Device: `COM9`
- Baudrate: `1000000`
- IDs: `11, 12, 13, 14, 15`

If your robot uses a different port, update `device_name` in `robot_adapter.py`.
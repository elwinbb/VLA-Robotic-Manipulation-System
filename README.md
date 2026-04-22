# Robot Vision Prompt Web Interface

This project includes a Flask web app that now uses the same RealSense + Gemini + calibration + robot-motion pipeline as `pick_and_place_gemini.ipynb`.

## What it does

- Streams aligned RealSense color frames in-browser (`/api/video_feed`)
- Runs the notebook Gemini model `gemini-robotics-er-1.6-preview` against the latest RealSense frame (`/api/vision/run`)
- Supports prompt types for:
  - `Description`
  - `Pick Object`
  - `Pick & Place`
- Shows the latest annotated prompt output image (`/api/vision/overlay`)
- Streams run state, structured output, and logs into the UI (`/api/vision/state`)
- Can optionally execute the notebook pickup / pick-and-place motion flow from the web UI

## Setup

```powershell
python -m pip install -r requirements.txt
```

You will need:

- A RealSense camera with `pyrealsense2`
- A valid `GEMINI_API_KEY`
- `dynamixel_sdk` plus the robot hardware if you want execution from the site

## Run

```powershell
$env:GEMINI_API_KEY="your_gemini_key_here"
python app.py
```

Open: `http://127.0.0.1:5000`

## Optional environment variables

```powershell
$env:GEMINI_API_KEY="your_gemini_key_here"
$env:GEMINI_MODEL="gemini-robotics-er-1.6-preview"
$env:OPENAI_API_KEY="your_key_here"
$env:OPENAI_MODEL="gpt-4.1-mini"
$env:PORT="5000"
```

## Camera troubleshooting

If the camera panel is offline:

1. Make sure the RealSense camera is connected and not in use by another app.
2. Use the `Restart pipeline` button in the UI.
3. Check health endpoint for diagnostics:

`http://127.0.0.1:5000/api/health`

Prompt analysis endpoints:

- `POST /api/vision/run` with JSON body `{ "prompt_type": "description", "prompt": "Describe the scene.", "execute_robot": false }`
- `GET /api/vision/state`
- `GET /api/vision/overlay`
- `POST /api/vision/clear`

## Legacy robot command examples

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

## Notes

- The website now uses the notebook pipeline directly instead of the earlier lightweight preview-only service.
- The notebook cell that hardcoded a Gemini key was not copied into the web app. The Flask app reads `GEMINI_API_KEY` from your environment instead.
- For `Pick Object` and `Pick & Place`, the UI can run preview only or execute the robot after ArUco calibration, matching the notebook flow.

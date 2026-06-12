# API Global — Internet access for the CSE498 model API

Exposes your local FastAPI server (in `../API`) to the public internet over HTTPS,
so your Android app can use the models from **anywhere** (mobile data, other WiFi, etc.).

## TL;DR

1. Sign up (free) at https://dashboard.ngrok.com/signup
2. Copy your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
3. Paste it into `ngrok_token.txt` (create the file in this folder)
4. Run:

   ```powershell
   cd "D:\CSE498 Research Updated\API Global"
   .\expose.ps1
   ```

5. Open the `https://*.ngrok-free.app/` URL in any browser to use the friendly
   web UI (pick a model from a dropdown, drop in a retinal image, get a
   prediction). The same URL works as the **Base URL** in the CSE498r Android
   app (include the trailing `/`).

   Useful paths on the same URL:
   - `/`      — friendly web UI (recommended for humans)
   - `/docs`  — interactive API reference (for developers)
   - `/models`— raw JSON list of available models

Read `instructions.txt` for the full walkthrough, troubleshooting, and how to
upgrade to a permanent URL or 24/7 hosting.

## Files

- `expose.ps1` — one-click: starts your API + opens public HTTPS tunnel
- `stop.ps1` — stops the tunnel and the local API server
- `ngrok_token.txt.example` — template for your ngrok authtoken
- `instructions.txt` — full step-by-step guide

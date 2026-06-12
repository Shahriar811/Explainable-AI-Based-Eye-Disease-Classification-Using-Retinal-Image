$ErrorActionPreference = "Stop"

function Get-PythonCmd {
  # Prefer Python 3.11 via Windows py launcher (best chance for torch wheels).
  if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
      py -3.11 -c "import sys; print(sys.version)" *> $null
      return "py -3.11"
    } catch {
      # ignore
    }
  }
  return "python"
}

$PY = Get-PythonCmd

if (!(Test-Path ".\.venv")) {
  & $PY -m venv .venv
}

.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\pip install -r requirements.txt

$env:PYTHONPATH = (Get-Location).Path
$HOST_ADDR = if ([string]::IsNullOrWhiteSpace($env:API_HOST)) { "127.0.0.1" } else { $env:API_HOST }
$PORT_NUM = if ([string]::IsNullOrWhiteSpace($env:API_PORT)) { 8000 } else { [int]$env:API_PORT }
.\.venv\Scripts\python -m uvicorn server.main:app --host $HOST_ADDR --port $PORT_NUM --reload


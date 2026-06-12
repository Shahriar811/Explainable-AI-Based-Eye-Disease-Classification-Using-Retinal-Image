$ErrorActionPreference = "SilentlyContinue"

Write-Host "Stopping ngrok..." -ForegroundColor Yellow
Get-Process ngrok -ErrorAction SilentlyContinue | Stop-Process -Force

Write-Host "Stopping any uvicorn workers bound to port 8000..." -ForegroundColor Yellow
# Kill the python process(es) that are serving uvicorn on port 8000.
$conns = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($conns) {
    foreach ($c in $conns) {
        try {
            $p = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
            if ($p) {
                Write-Host "  Killing PID $($p.Id) ($($p.ProcessName))"
                Stop-Process -Id $p.Id -Force
            }
        } catch { }
    }
} else {
    Write-Host "  Nothing was listening on port 8000."
}

Write-Host "Done." -ForegroundColor Green

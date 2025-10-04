param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    if ($Clean) {
        if (Test-Path "build") {
            Remove-Item "build" -Recurse -Force
        }
        if (Test-Path "dist") {
            Remove-Item "dist" -Recurse -Force
        }
    }

    python scripts\build_msi.py bdist_msi
}
finally {
    Pop-Location
}

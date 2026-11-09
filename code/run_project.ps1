param(
    [string]$Python = (Join-Path $PSScriptRoot '..\..\.venv\Scripts\python.exe'),
    [switch]$Resume
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$arguments = @('run_project.py')
if ($Resume) { $arguments += '--resume' }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Project pipeline failed; see ..\results\execution.log' }

param(
    [Parameter(Mandatory = $true)]
    [string]$Source
)

$ErrorActionPreference = "Stop"

$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$sourceName = Split-Path -Leaf $sourcePath

if ($sourceName -ne "dataset_streamlit_shell") {
    throw "Source must be a dataset_streamlit_shell folder. Got: $sourcePath"
}

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$templateRoot = Join-Path $projectRoot "src\add_dataset_streamlit_shell\templates"
$destination = Join-Path $templateRoot "dataset_streamlit_shell"

if (-not (Test-Path -LiteralPath (Join-Path $sourcePath "app.py"))) {
    throw "Source is missing app.py: $sourcePath"
}

if (-not (Test-Path -LiteralPath (Join-Path $sourcePath "data_ui.py"))) {
    throw "Source is missing data_ui.py: $sourcePath"
}

if (Test-Path -LiteralPath $destination) {
    Remove-Item -LiteralPath $destination -Recurse -Force
}

New-Item -ItemType Directory -Path $templateRoot -Force | Out-Null
Copy-Item -LiteralPath $sourcePath -Destination $destination -Recurse -Force

# Keep the template clean: ship folders and placeholders, not local runtime data.
Get-ChildItem -LiteralPath (Join-Path $destination "data") -Filter "*.csv" -File -ErrorAction SilentlyContinue |
    Remove-Item -Force

Get-ChildItem -LiteralPath (Join-Path $destination "data") -Filter "*.jsonl" -File -ErrorAction SilentlyContinue |
    Remove-Item -Force

Get-ChildItem -LiteralPath (Join-Path $destination "sessions") -Filter "*.jsonl" -File -ErrorAction SilentlyContinue |
    Remove-Item -Force

if (Test-Path -LiteralPath (Join-Path $destination "uploads")) {
    Remove-Item -LiteralPath (Join-Path $destination "uploads") -Recurse -Force
}

foreach ($folder in @("data", "sessions")) {
    $folderPath = Join-Path $destination $folder
    if (-not (Test-Path -LiteralPath $folderPath)) {
        New-Item -ItemType Directory -Path $folderPath -Force | Out-Null
    }

    $gitkeep = Join-Path $folderPath ".gitkeep"
    if (-not (Test-Path -LiteralPath $gitkeep)) {
        New-Item -ItemType File -Path $gitkeep -Force | Out-Null
    }
}

Write-Host "Synced dataset_streamlit_shell template."
Write-Host "Source:      $sourcePath"
Write-Host "Destination: $destination"

$rulesDir = ".roo/rules"
$outputFile = "GEMINI.md"

if (-not (Test-Path $rulesDir)) {
    Write-Error "Directory $rulesDir not found."
    exit 1
}

$content = @()
$content += "# Project Rules (Synced from .roo/rules)"
$content += ""

# Get all .md files, excluding those starting with a dot
$files = Get-ChildItem -Path $rulesDir -Filter "*.md" | Where-Object { $_.Name -notlike ".*" } | Sort-Object Name

foreach ($file in $files) {
    $relativeLink = "$rulesDir/$($file.Name)"
    $content += "## [$($file.BaseName)]($relativeLink)"
    $content += ""
    $fileContent = Get-Content -Path $file.FullName -Raw
    $content += $fileContent
    $content += ""
    $content += "---"
    $content += ""
}

$content | Out-File -FilePath $outputFile -Encoding utf8
Write-Host "Successfully generated $outputFile from $rulesDir"

param(
    [Parameter(Mandatory = $true)]
    [string]$IsaacSimPath
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$target = (Resolve-Path -LiteralPath $IsaacSimPath).Path.TrimEnd('\')
$link = Join-Path $projectRoot 'runtime'
if (-not (Test-Path -LiteralPath (Join-Path $target 'python.bat') -PathType Leaf) -or
    -not (Test-Path -LiteralPath (Join-Path $target 'kit') -PathType Container)) {
    throw 'Choose the extracted Isaac Sim installation containing python.bat and kit/.'
}
if ($target.Equals($link, [StringComparison]::OrdinalIgnoreCase) -or
    $target.StartsWith($link + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The runtime link cannot point to itself or a directory beneath itself.'
}
$existing = Get-Item -Force -LiteralPath $link -ErrorAction SilentlyContinue
if ($existing) {
    if (-not ($existing.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'runtime is a real directory. Move it outside the repository before linking.'
    }
    # Delete only the link itself; never recurse into the installation.
    [IO.Directory]::Delete($link)
}
try {
    New-Item -ItemType Junction -Path $link -Target $target | Out-Null
} catch {
    if ($existing -and $existing.Target) {
        New-Item -ItemType Junction -Path $link -Target @($existing.Target)[0] | Out-Null
    }
    throw
}
Write-Host "Isaac Sim runtime linked to $target"
if (Test-Path -LiteralPath (Join-Path $target 'VERSION')) {
    Write-Host "Version: $(Get-Content -LiteralPath (Join-Path $target 'VERSION') -Raw)"
}

param(
    [Parameter(Mandatory=$true, Position=0)]
    [int]$MaxBytes,
    [Parameter(Mandatory=$true, Position=1, ValueFromRemainingArguments=$true)]
    [string[]]$CmdAndArgs
)

if ($CmdAndArgs.Count -eq 0) {
    Write-Error "Usage: .\cap_output.ps1 <maxBytes> <command> [args...]"
    exit 2
}

$cmd = $CmdAndArgs[0]
$args = @()
if ($CmdAndArgs.Count -gt 1) { $args = $CmdAndArgs[1..($CmdAndArgs.Count-1)] }

# Run the command and collect output as a single string
try {
    $output = & $cmd @args 2>&1 | Out-String
} catch {
    $output = $_.Exception.Message + "`n"
}

if ($null -eq $output) { $output = "" }

if ($output.Length -gt $MaxBytes) {
    $output = $output.Substring(0, $MaxBytes)
}

Write-Output $output

param(
    [string]$PythonPath = 'D:\anaconda3\envs\torchnew\python.exe',
    [string[]]$Models = @(
        'pointmae',
        'pointtransv3',
        'pointtransformer',
        'upp',
        'pointtransv2',
        'pointnet2msg'
    )
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$trainScript = Join-Path $PSScriptRoot 'train.py'
$logDirectory = Join-Path $projectRoot 'logs\CLS'
$queueTimestamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffffff'
$queueLog = Join-Path $logDirectory "training_queue_$queueTimestamp.log"
$statePath = Join-Path $logDirectory "training_queue_$queueTimestamp.state.csv"

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python executable does not exist: $PythonPath"
}
if (-not (Test-Path -LiteralPath $trainScript)) {
    throw "Training script does not exist: $trainScript"
}

New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
Set-Location -LiteralPath $projectRoot

function Write-QueueLog {
    param([string]$Message)

    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Message"
    Add-Content -LiteralPath $queueLog -Value $line -Encoding utf8
    Write-Output $line
}

'model,status,start_time,end_time,exit_code,stdout_log,stderr_log' |
    Set-Content -LiteralPath $statePath -Encoding utf8

Write-QueueLog "queue_started models=$($Models -join ',')"
Write-QueueLog 'config epochs=100 batch_size=32 num_aug=3 amp=False tf32=False'

foreach ($model in $Models) {
    $taskTimestamp = Get-Date -Format 'yyyyMMdd_HHmmss_ffffff'
    $stdoutLog = Join-Path $logDirectory "queue_${model}_$taskTimestamp.stdout.log"
    $stderrLog = Join-Path $logDirectory "queue_${model}_$taskTimestamp.stderr.log"
    $startTime = Get-Date
    $arguments = @(
        $trainScript,
        '--model', $model,
        '--batch-size', '32',
        '--epochs', '100',
        '--num-aug', '3',
        '--box-head', 'centroid',
        '--seg-loss-weight', '0.5',
        '--no-amp',
        '--no-tf32'
    )

    Write-QueueLog "task_started model=$model"
    $processParameters = @{
        FilePath = $PythonPath
        ArgumentList = $arguments
        WorkingDirectory = $projectRoot
        RedirectStandardOutput = $stdoutLog
        RedirectStandardError = $stderrLog
        WindowStyle = 'Hidden'
        Wait = $true
        PassThru = $true
    }
    $process = Start-Process @processParameters

    $endTime = Get-Date
    $status = if ($process.ExitCode -eq 0) { 'completed' } else { 'failed' }
    $stateRow = @(
        $model,
        $status,
        $startTime.ToString('o'),
        $endTime.ToString('o'),
        $process.ExitCode,
        $stdoutLog,
        $stderrLog
    ) -join ','
    Add-Content -LiteralPath $statePath -Value $stateRow -Encoding utf8
    Write-QueueLog "task_$status model=$model exit_code=$($process.ExitCode) duration=$($endTime - $startTime)"

    if ($process.ExitCode -ne 0) {
        Write-QueueLog "queue_stopped failed_model=$model"
        exit $process.ExitCode
    }
}

Write-QueueLog 'queue_completed'

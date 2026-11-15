$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$releaseRoot = Join-Path $projectRoot '.runtime\evaluation-desktop-20261105-r1'
$taskName = 'RoadModelEvaluationAgent-20261105-r1'
$sourceConfig = Join-Path $projectRoot 'evidence\evaluation-20261105\agent-config.json'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    throw 'The evaluation agent task already exists. Inspect it before changing the installation.'
}
if (Test-Path -LiteralPath $releaseRoot) {
    throw 'The desktop release already exists. Preserve its configuration and receipts.'
}
$pythonBinary = (& python -c 'import sys;print(sys.executable)').Trim()
$pythonGui = Join-Path (Split-Path $pythonBinary) 'pythonw.exe'
if (!(Test-Path -LiteralPath $pythonGui)) { throw 'The installed Python GUI host is unavailable' }
New-Item -ItemType Directory -Path $releaseRoot | Out-Null
foreach ($folder in @('pcm_eval','tools','configs','examples')) {
    $destination = Join-Path $releaseRoot $folder
    New-Item -ItemType Directory -Path $destination | Out-Null
    if ($folder -eq 'pcm_eval') {
        Get-ChildItem -LiteralPath (Join-Path $projectRoot $folder) -Filter '*.py' -File | Copy-Item -Destination $destination
    } elseif ($folder -eq 'examples') {
        Get-ChildItem -LiteralPath (Join-Path $projectRoot $folder) -Filter '*.json' -File | Copy-Item -Destination $destination
    }
}
foreach ($name in @('evaluation_agent.py','delivery_demo_agent.py','model_space_desktop.py')) {
    Copy-Item -LiteralPath (Join-Path $projectRoot "tools\$name") -Destination (Join-Path $releaseRoot "tools\$name")
}
Copy-Item -LiteralPath (Join-Path $projectRoot 'configs\delivery_catalog.json') -Destination (Join-Path $releaseRoot 'configs\delivery_catalog.json')
foreach ($relative in @(
    'evidence\model-evaluation\model-reference-20261027\model-input-binding.json',
    'evidence\model-evaluation\model-reference-20261027\native-gt-preparation.json',
    'evidence\requested-dataset\data_link_26-10-20-1_train859_eval135_v7.json'
)) {
    $destination = Join-Path $releaseRoot $relative
    New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $projectRoot $relative) -Destination $destination
}
Copy-Item -LiteralPath $sourceConfig -Destination (Join-Path $releaseRoot 'configs\agent.json')
$arguments = '-B "' + (Join-Path $releaseRoot 'tools\evaluation_agent.py') + '" --config "' + (Join-Path $releaseRoot 'configs\agent.json') + '"'
$action = New-ScheduledTaskAction -Execute $pythonGui -Argument $arguments -WorkingDirectory $releaseRoot
$logon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Days 365)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -Hidden
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$task = New-ScheduledTask -Action $action -Trigger @($logon,$repeat) -Settings $settings -Principal $principal
Register-ScheduledTask -TaskName $taskName -InputObject $task | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$releaseRoot = Join-Path $projectRoot '.runtime\evaluation-environment-desktop-20261106-r3'
$taskName = 'RoadModelEvaluationAgent-20261106-r3'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Inspect existing r3 task before changing it' }
if (Test-Path -LiteralPath $releaseRoot) { throw 'Preserve existing pinned desktop release' }
$pythonBinary = (& python -c 'import sys;print(sys.executable)').Trim()
$pythonGui = Join-Path (Split-Path $pythonBinary) 'pythonw.exe'
if (!(Test-Path -LiteralPath $pythonGui)) { throw 'Installed Python GUI host unavailable' }
New-Item -ItemType Directory -Path $releaseRoot | Out-Null
foreach ($folder in @('pcm_eval','examples')) {
    $dest = Join-Path $releaseRoot $folder
    New-Item -ItemType Directory -Path $dest | Out-Null
    $pattern = if ($folder -eq 'pcm_eval') { '*.py' } else { '*.json' }
    Get-ChildItem -LiteralPath (Join-Path $projectRoot $folder) -Filter $pattern -File | Copy-Item -Destination $dest
}
foreach ($relative in @(
    'tools\evaluation_agent.py','tools\evaluation_space_actor.py','tools\delivery_demo_agent.py','tools\model_space_desktop.py',
    'configs\delivery_catalog.json','schemas\space-preparation-result.schema.json','skills\roadmodel-space-preparation\SKILL.md',
    'evidence\model-evaluation\model-reference-20261027\model-input-binding.json',
    'evidence\model-evaluation\model-reference-20261027\native-gt-preparation.json',
    'evidence\requested-dataset\data_link_26-10-20-1_train859_eval135_v7.json'
)) {
    $destination = Join-Path $releaseRoot $relative
    New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $projectRoot $relative) -Destination $destination
}
Copy-Item -LiteralPath (Join-Path $projectRoot 'evidence\environment-init-20261106\r3-agent-config.json') -Destination (Join-Path $releaseRoot 'configs\agent.json')
$arguments = '-B "' + (Join-Path $releaseRoot 'tools\evaluation_agent.py') + '" --config "' + (Join-Path $releaseRoot 'configs\agent.json') + '"'
$action = New-ScheduledTaskAction -Execute $pythonGui -Argument $arguments -WorkingDirectory $releaseRoot
$logon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Days 365)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -Hidden
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -InputObject (New-ScheduledTask -Action $action -Trigger @($logon,$repeat) -Settings $settings -Principal $principal) | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State

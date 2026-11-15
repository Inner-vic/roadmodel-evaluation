$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$releaseRoot = Join-Path $repoRoot '.runtime\space-preparation-20261103-r3'
$configSource = Join-Path $repoRoot 'configs\model_space_desktop.json'
$taskName = 'RoadModelSpacePreparation-20261103-r3'
$pythonBinary = (& python -c 'import sys;print(sys.executable)').Trim()
$pythonGui = Join-Path (Split-Path $pythonBinary) 'pythonw.exe'
if (!(Test-Path -LiteralPath $pythonGui)) { throw 'The existing installed Python GUI host is unavailable' }
if (Test-Path -LiteralPath $releaseRoot) { throw 'Desktop release exists; inspect before replacing it' }
New-Item -ItemType Directory -Path $releaseRoot | Out-Null
foreach ($folder in @('pcm_eval','schemas','skills')) {
    $destination = Join-Path $releaseRoot $folder
    New-Item -ItemType Directory -Path $destination | Out-Null
    if ($folder -eq 'pcm_eval') {
        Get-ChildItem -LiteralPath (Join-Path $repoRoot $folder) -Filter '*.py' -File | Copy-Item -Destination $destination
    } elseif ($folder -eq 'schemas') {
        Copy-Item -LiteralPath (Join-Path $repoRoot 'schemas\space-preparation-result.schema.json') -Destination $destination
    } else {
        New-Item -ItemType Directory -Path (Join-Path $destination 'roadmodel-space-preparation') | Out-Null
        Copy-Item -LiteralPath (Join-Path $repoRoot 'skills\roadmodel-space-preparation\SKILL.md') -Destination (Join-Path $destination 'roadmodel-space-preparation\SKILL.md')
    }
}
New-Item -ItemType Directory -Path (Join-Path $releaseRoot 'tools') | Out-Null
New-Item -ItemType Directory -Path (Join-Path $releaseRoot 'configs') | Out-Null
Copy-Item -LiteralPath (Join-Path $repoRoot 'tools\model_space_desktop.py') -Destination (Join-Path $releaseRoot 'tools\model_space_desktop.py')
Copy-Item -LiteralPath $configSource -Destination (Join-Path $releaseRoot 'configs\model_space_desktop.json')
$arguments = '-B "' + (Join-Path $releaseRoot 'tools\model_space_desktop.py') + '" --config "' + (Join-Path $releaseRoot 'configs\model_space_desktop.json') + '"'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) { throw 'The named scheduled task already exists; do not overwrite an unknown host' }
$previous = Get-ScheduledTask -TaskName 'RoadModelSpacePreparation-20261103-r2' -ErrorAction SilentlyContinue
if ($previous) {
    $expectedScript = Join-Path $repoRoot '.runtime\space-preparation-20261103-r2\tools\model_space_desktop.py'
    if ($previous.Actions.Count -ne 1 -or !$previous.Actions[0].Arguments.Contains($expectedScript)) {
        throw 'Previous host identity differs; do not stop an unrelated task'
    }
    Export-ScheduledTask -TaskName $previous.TaskName | Set-Content -LiteralPath (Join-Path $releaseRoot 'previous-host.xml')
    Disable-ScheduledTask -TaskName $previous.TaskName | Out-Null
    Stop-ScheduledTask -TaskName $previous.TaskName
}
$action = New-ScheduledTaskAction -Execute $pythonGui -Argument $arguments -WorkingDirectory $releaseRoot
$logon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -Hidden
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$task = New-ScheduledTask -Action $action -Trigger @($logon,$repeat) -Settings $settings -Principal $principal
Register-ScheduledTask -TaskName $taskName -InputObject $task | Out-Null
Start-ScheduledTask -TaskName $taskName
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State

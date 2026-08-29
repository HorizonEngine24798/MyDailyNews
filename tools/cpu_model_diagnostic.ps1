param(
    [string]$BaseUrl = "http://127.0.0.1:8081/v1",
    [string]$Model = "Qwen3-1.7B-Q4_K_M",
    [int]$TimeoutSeconds = 180,
    [string]$OutputRoot = "output/diagnostics/cpu-model-test"
)

$ErrorActionPreference = "Stop"
$base = $BaseUrl.TrimEnd("/")
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDirectory = Join-Path $OutputRoot $stamp
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

function Get-ServerProcesses {
    @(Get-Process llama-server -ErrorAction SilentlyContinue)
}

function Measure-Request {
    param([string]$Label, [hashtable]$Payload)

    $logicalProcessors = [Math]::Max(1, [Environment]::ProcessorCount)
    $processesBefore = Get-ServerProcesses
    $cpuBefore = ($processesBefore | Measure-Object -Property CPU -Sum).Sum
    if ($null -eq $cpuBefore) { $cpuBefore = 0.0 }
    $workingSetBefore = ($processesBefore | Measure-Object -Property WorkingSet64 -Sum).Sum
    if ($null -eq $workingSetBefore) { $workingSetBefore = 0 }

    $watch = [System.Diagnostics.Stopwatch]::StartNew()
    $response = Invoke-RestMethod -Method Post -Uri "$base/chat/completions" -ContentType "application/json" -Body ($Payload | ConvertTo-Json -Depth 10) -TimeoutSec $TimeoutSeconds
    $watch.Stop()

    $processesAfter = Get-ServerProcesses
    $cpuAfter = ($processesAfter | Measure-Object -Property CPU -Sum).Sum
    if ($null -eq $cpuAfter) { $cpuAfter = 0.0 }
    $workingSetAfter = ($processesAfter | Measure-Object -Property WorkingSet64 -Sum).Sum
    if ($null -eq $workingSetAfter) { $workingSetAfter = 0 }

    $timings = if ($response.timings) { $response.timings } else { @{} }
    $usage = if ($response.usage) { $response.usage } else { @{} }
    $elapsed = [Math]::Round($watch.Elapsed.TotalSeconds, 3)
    $evalTokensValue = if ($null -ne $timings.predicted_n) { $timings.predicted_n } elseif ($null -ne $usage.completion_tokens) { $usage.completion_tokens } else { 0 }
    $evalMsValue = if ($null -ne $timings.predicted_ms) { $timings.predicted_ms } else { 0 }
    $evalTokens = [double]$evalTokensValue
    $evalMs = [double]$evalMsValue
    $tokensPerSecond = if ($evalMs -gt 0) { [Math]::Round(($evalTokens / $evalMs) * 1000, 3) } else { $null }
    $cpuPercent = if ($elapsed -gt 0) { [Math]::Round((($cpuAfter - $cpuBefore) / $elapsed / $logicalProcessors) * 100, 2) } else { 0 }

    $content = $response.choices[0].message.content
    if ([string]::IsNullOrWhiteSpace($content) -and $response.choices[0].message.reasoning_content) {
        $content = $response.choices[0].message.reasoning_content
    }
    $validJson = $true
    try { $null = $content | ConvertFrom-Json -ErrorAction Stop } catch { $validJson = $false }

    [pscustomobject]@{
        label = $Label
        elapsed_seconds = $elapsed
        output_tokens = $usage.completion_tokens
        prompt_tokens = $usage.prompt_tokens
        tokens_per_second = $tokensPerSecond
        server_cpu_percent_of_machine = $cpuPercent
        server_working_set_mb_before = [Math]::Round($workingSetBefore / 1MB, 1)
        server_working_set_mb_after = [Math]::Round($workingSetAfter / 1MB, 1)
        finish_reason = $response.choices[0].finish_reason
        response_json_valid = $validJson
        response = $content
        timings = $timings
    }
}

$health = Invoke-RestMethod -Method Get -Uri "$base/models" -TimeoutSec 15
$system = "Return only valid JSON. Do not use markdown or explain your reasoning."
$cases = @(
    @{ label = "warmup"; user = 'Return exactly this JSON object: {"ok":true}'; max_tokens = 24 },
    @{ label = "headline_selection"; user = 'Given these headlines, return JSON with keys selected_ids (array of strings) and reason (string under 20 words). Headlines: [{"id":"a","title":"Central bank holds rates"},{"id":"b","title":"Local parade this weekend"},{"id":"c","title":"Ceasefire talks resume"}]. Select the two most globally significant.'; max_tokens = 96 },
    @{ label = "brief_summary"; user = 'Return JSON with keys lead (one sentence) and bullets (exactly two strings). Source facts: inflation fell from 3.1% to 2.8%; the central bank kept rates unchanged; officials said future cuts depend on data.'; max_tokens = 120 }
)

$results = foreach ($case in $cases) {
    $payload = @{
        model = $Model
        messages = @(
            @{ role = "system"; content = $system },
            @{ role = "user"; content = $case.user }
        )
        temperature = 0.0
        top_p = 1.0
        max_tokens = $case.max_tokens
        response_format = @{ type = "json_object" }
    }
    Measure-Request -Label $case.label -Payload $payload
}

$report = [pscustomobject]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    base_url = $base
    model = $Model
    server_models = $health.data
    results = $results
}
$reportPath = Join-Path $outputDirectory "report.json"
$report | ConvertTo-Json -Depth 12 | Set-Content -Encoding UTF8 $reportPath
$report | ConvertTo-Json -Depth 12
Write-Host "Diagnostic report: $reportPath"

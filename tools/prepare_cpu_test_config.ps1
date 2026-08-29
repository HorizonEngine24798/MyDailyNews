param(
    [string]$SourceConfig = "config.local.json",
    [string]$DestinationConfig = "config.cpu-qwen1.7b.json",
    [string]$BaseUrl = "http://127.0.0.1:8081/v1",
    [switch]$FullPipeline
)

$ErrorActionPreference = "Stop"
function Set-JsonProperty {
    param($Object, [string]$Name, $Value)
    $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
}

if (-not (Test-Path -LiteralPath $SourceConfig)) {
    throw "Source config not found: $SourceConfig"
}
if (Test-Path -LiteralPath $DestinationConfig) {
    throw "Refusing to overwrite existing CPU test config: $DestinationConfig"
}

$config = Get-Content -LiteralPath $SourceConfig -Raw | ConvertFrom-Json
$config.output_dir = "output+cpu-test"

foreach ($sectionName in @("ai_summary", "ai_final")) {
    $ai = $config.$sectionName
    Set-JsonProperty $ai "backend" "llama_cpp_server"
    Set-JsonProperty $ai "model_id" "Qwen3-1.7B-Q4_K_M"
    Set-JsonProperty $ai "server_model" "Qwen3-1.7B-Q4_K_M"
    Set-JsonProperty $ai "base_url" $BaseUrl
    Set-JsonProperty $ai "context_window_tokens" 2048
    Set-JsonProperty $ai "max_input_tokens" 1400
    Set-JsonProperty $ai "max_new_tokens" 256
    Set-JsonProperty $ai "request_timeout_seconds" 180
    Set-JsonProperty $ai "token_estimation_chars_per_token" 4.0
    Set-JsonProperty $ai "response_format" "json_schema"
    Set-JsonProperty $ai "enable_thinking" $false
    Set-JsonProperty $ai "manage_server" $false
    Set-JsonProperty $ai "server_executable" ""
    Set-JsonProperty $ai "server_model_path" ""
    Set-JsonProperty $ai "server_arguments" @()
}

foreach ($sectionName in @("general_filtering", "filtering")) {
    $filter = $config.$sectionName
    $filter.max_candidates_for_ai = 12
    $filter.max_headlines_per_ai_batch = 3
    $filter.headline_max_input_tokens = 1200
    $filter.headline_max_new_tokens = 256
    $filter.headline_single_replay_max_new_tokens = 128
    $filter.max_selected_articles = 3
    $filter.article_text_max_chars = 1500
}

$config.cache.dir = ".cache/cpu-model-test"
$config.enrichment.enabled = $FullPipeline.IsPresent
$config.enrichment.mode = if ($FullPipeline) { "story_llm" } else { "disabled" }
$config.enrichment.max_story_threads = 2
$config.enrichment.planner_max_questions_per_story = 1
$config.enrichment.planner_max_input_tokens = 1000
$config.enrichment.planner_max_new_tokens = 192
$config.enrichment.search_results_per_query = 3
$config.enrichment.max_fetched_research_pages_per_story = 1
$config.enrichment.max_selected_article_excerpt_chars = 800
$config.enrichment.max_research_excerpt_chars = 600
$config.enrichment.synthesis_max_input_tokens = 1200
$config.enrichment.synthesis_max_new_tokens = 256
$config.narrative_briefing.enabled = $FullPipeline.IsPresent
$config.narrative_briefing.max_input_tokens = 1400
$config.narrative_briefing.max_new_tokens = 256
$config.narrative_briefing.target_words = 250
$config.tts.enabled = $false
$config.pipeline.default_series = if ($FullPipeline) { @("briefs", "enrichment", "perspectives_report", "narrative_brief") } else { @("briefs") }
if ($config.memory) {
    $config.memory.state_dir = "state/cpu-model-test-memory"
}

$config | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $DestinationConfig -Encoding UTF8
Write-Host "Created isolated CPU test config: $DestinationConfig"

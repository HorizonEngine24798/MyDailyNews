import { api, clone } from "./api.js";
import { byId, setStatus } from "./dom.js";
import { renderForm } from "./forms.js";
import { bindMemoryRepairEvents, renderMemory } from "./memory.js";
import {
  renderReportList,
  selectReport,
  setReportMemoryReload,
  sortedReports,
  updateReportTypeOptions,
} from "./reports.js";
import { bindRunEvents, loadRuns, renderRuns, setRunRefreshCallbacks } from "./runs.js";
import { defaultStoryIndex, state } from "./state.js";

const PIPELINE_MODULES = [
  ["briefs", "Briefs"],
  ["enrichment", "Enrichment"],
  ["narrative_brief", "Narrative brief"],
  ["tts", "TTS audio"],
];
const DEFAULT_PIPELINE_SERIES = ["briefs", "enrichment", "narrative_brief"];

async function loadInitial() {
  try {
    state.app = await api("/api/state");
    byId("configPath").textContent = state.app.config_path || "";
    await Promise.all([loadConfig(), loadReports(), loadMemory(), loadLearned(), loadRuns()]);
    setStatus("Ready");
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function loadConfig() {
  state.config = await api("/api/config");
  state.configDraft = clone(state.config.config || {});
  state.userMemoryDraft = state.configDraft.user_memory || {};
  renderConfigForms();
}

async function loadReports() {
  const payload = await api("/api/reports");
  state.reports = payload.reports || [];
  updateReportTypeOptions();
  renderReportList();

  if (!state.currentReport && state.reports.length) {
    const first = sortedReports(state.reports)[0];
    await selectReport(first.id);
  } else if (!state.reports.length) {
    byId("reportHeader").innerHTML = "";
    byId("feedbackPanel").innerHTML = "";
    byId("markdownView").innerHTML = `<div class="empty">No reports found.</div>`;
  }
}

async function loadMemory() {
  state.memory = await api("/api/memory");
  state.storyIndexDraft = clone(state.memory.story_index_file || defaultStoryIndex());
  renderMemory();
}

async function loadLearned() {
  state.learned = await api("/api/learned-preferences");
  state.learnedDraft = clone(state.learned.preferences || {});
  renderLearnedForm();
}

async function refreshCurrent() {
  setStatus("Refreshing");
  try {
    if (state.view === "reports") {
      await loadReports();
    } else if (state.view === "settings") {
      await loadConfig();
    } else if (state.view === "profiles") {
      await Promise.all([loadConfig(), loadLearned()]);
    } else if (state.view === "memory") {
      await Promise.all([loadMemory(), loadLearned()]);
    } else if (state.view === "runs") {
      await loadRuns();
    }
    setStatus("Ready");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function setView(view) {
  state.view = view;
  document.querySelectorAll(".view").forEach((node) => {
    node.classList.toggle("active", node.id === `${view}View`);
  });
  document.querySelectorAll(".nav-button").forEach((node) => {
    node.classList.toggle("active", node.dataset.view === view);
  });
  byId("viewTitle").textContent =
    {
      reports: "Reports",
      settings: "Settings",
      profiles: "Profiles",
      memory: "Memory",
      runs: "Runs",
    }[view] || "MyDailyNews";
  if (view === "runs") {
    renderRuns();
  }
}

function renderConfigForms() {
  state.configDraft.user_memory = state.configDraft.user_memory || {};
  state.configDraft.sources = state.configDraft.sources || {};
  state.configDraft.general_topics = state.configDraft.general_topics || [];
  state.configDraft.topics_to_examine = state.configDraft.topics_to_examine || [];
  state.configDraft.general_filtering = state.configDraft.general_filtering || {};
  state.configDraft.filtering = state.configDraft.filtering || {};
  state.configDraft.pipeline = state.configDraft.pipeline || {};
  state.userMemoryDraft = state.configDraft.user_memory;
  renderForm("userMemoryForm", () => state.userMemoryDraft);
  renderForm("settingsSourcesForm", () => state.configDraft.sources);
  renderForm("settingsGeneralTopicsForm", () => state.configDraft.general_topics);
  renderForm("settingsDetailedTopicsForm", () => state.configDraft.topics_to_examine);
  renderForm("settingsGeneralFilteringForm", () => state.configDraft.general_filtering);
  renderForm("settingsFilteringForm", () => state.configDraft.filtering);
  renderPipelineForm();
}

function renderLearnedForm() {
  renderForm("learnedForm", () => state.learnedDraft);
  renderForm("memoryLearnedForm", () => state.learnedDraft);
}

function renderPipelineForm() {
  const host = byId("settingsPipelineForm");
  if (!Array.isArray(state.configDraft.pipeline.default_series)) {
    state.configDraft.pipeline.default_series = [...DEFAULT_PIPELINE_SERIES];
  }
  const selected = new Set(state.configDraft.pipeline.default_series);
  host.innerHTML = `<div class="form-grid">${PIPELINE_MODULES.map(([module, label]) => pipelineToggle(module, label, selected)).join("")}</div>`;
  host.querySelectorAll("[data-pipeline-module]").forEach((input) => {
    input.addEventListener("change", () => {
      state.configDraft.pipeline.default_series = PIPELINE_MODULES.filter(([module]) => {
        return byId(`pipelineModule-${module}`).value === "true";
      }).map(([module]) => module);
    });
  });
}

function pipelineToggle(module, label, selected) {
  const value = selected.has(module) ? "true" : "false";
  return `
    <label class="field-row boolean-row">
      <span class="field-label">${label}</span>
      <select id="pipelineModule-${module}" class="field-input boolean-input" data-pipeline-module="${module}">
        <option value="true"${value === "true" ? " selected" : ""}>true</option>
        <option value="false"${value === "false" ? " selected" : ""}>false</option>
      </select>
    </label>
  `;
}

async function saveUserMemory() {
  await saveConfigSection("user_memory", state.userMemoryDraft, "Ground truth profile");
}

async function saveConfigSection(section, payload, label) {
  try {
    state.config = await api(`/api/config/section/${encodeURIComponent(section)}`, {
      method: "PUT",
      body: JSON.stringify(payload),
    });
    state.configDraft = clone(state.config.config || {});
    state.userMemoryDraft = state.configDraft.user_memory || {};
    renderConfigForms();
    setStatus(`${label} saved`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function saveLearned() {
  try {
    state.learned = await api("/api/learned-preferences", { method: "PUT", body: JSON.stringify(state.learnedDraft) });
    state.learnedDraft = clone(state.learned.preferences || {});
    renderLearnedForm();
    setStatus("Learned preferences saved");
    await loadMemory();
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function previewLearned(targetId = "learnedPreview") {
  try {
    const payload = await api("/api/previews/learned-preferences", {
      method: "POST",
      body: JSON.stringify(state.learnedDraft || {}),
    });
    const topics = (payload.effective_weights?.topics || []).map((row) => `${row.name}: ${row.weight}`);
    const sources = (payload.effective_weights?.sources || []).map((row) => `${row.name}: ${row.weight}`);
    const lines = [
      "Preferred topics: " + (payload.preferences?.preferred_topics || []).join(", "),
      "Avoided topics: " + (payload.preferences?.avoided_topics || []).join(", "),
      "Preferred sources: " + (payload.preferences?.preferred_sources || []).join(", "),
      "Avoided sources: " + (payload.preferences?.avoided_sources || []).join(", "),
      "",
      "Topic weights:",
      topics.length ? topics.join("\n") : "none",
      "",
      "Source weights:",
      sources.length ? sources.join("\n") : "none",
    ];
    byId(targetId).textContent = lines.join("\n");
    setStatus("Learned preferences preview updated");
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function saveStoryIndex() {
  try {
    state.memory = await api("/api/memory/story-index", { method: "PUT", body: JSON.stringify(state.storyIndexDraft) });
    state.storyIndexDraft = clone(state.memory.story_index_file || defaultStoryIndex());
    renderMemory();
    setStatus("Story index saved");
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function pruneMemory() {
  try {
    const payload = await api("/api/memory/prune", { method: "POST", body: JSON.stringify({}) });
    state.memory = payload.memory;
    state.storyIndexDraft = clone(state.memory.story_index_file || defaultStoryIndex());
    renderMemory();
    setStatus("Memory pruned");
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function runAutoconfig() {
  const output = byId("autoconfigOutput");
  output.textContent = "Running...";
  try {
    const payload = await api("/api/autoconfig", {
      method: "POST",
      body: JSON.stringify({ write: byId("autoconfigWrite").value }),
    });
    output.textContent = [
      `returncode: ${payload.returncode}`,
      `write_path: ${payload.write_path}`,
      "",
      payload.stdout || "",
      payload.stderr ? `stderr:\n${payload.stderr}` : "",
    ].join("\n");
    setStatus(payload.returncode === 0 ? "Autoconfig finished" : "Autoconfig failed", payload.returncode !== 0);
  } catch (error) {
    output.textContent = error.message;
    setStatus(error.message, true);
  }
}

function togglePane(which) {
  if (which === "content") {
    state.contentCollapsed = !state.contentCollapsed;
  } else {
    state.browserCollapsed = !state.browserCollapsed;
  }
  togglePaneClasses();
}

function togglePaneClasses() {
  byId("reportsLayout").classList.toggle("content-collapsed", state.contentCollapsed);
  byId("reportsLayout").classList.toggle("browser-collapsed", state.browserCollapsed);
  byId("reportContentPane").classList.toggle("collapsed", state.contentCollapsed);
  byId("reportBrowserPane").classList.toggle("collapsed", state.browserCollapsed);
  updatePaneButton("toggleReportContent", state.contentCollapsed, "report");
  updatePaneButton("toggleReportBrowser", state.browserCollapsed, "reports");
}

function updatePaneButton(id, collapsed, label) {
  const button = byId(id);
  const action = collapsed ? "Open" : "Close";
  button.setAttribute("aria-label", `${action} ${label}`);
  button.setAttribute("aria-expanded", collapsed ? "false" : "true");
  button.title = `${action} ${label}`;
}

function bindEvents() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", () => setView(button.dataset.view));
  });
  byId("refreshButton").addEventListener("click", refreshCurrent);
  byId("saveUserMemoryButton").addEventListener("click", saveUserMemory);
  byId("saveSettingsSourcesButton").addEventListener("click", () =>
    saveConfigSection("sources", state.configDraft.sources, "RSS sources")
  );
  byId("saveSettingsGeneralTopicsButton").addEventListener("click", () =>
    saveConfigSection("general_topics", state.configDraft.general_topics, "General topics")
  );
  byId("saveSettingsDetailedTopicsButton").addEventListener("click", () =>
    saveConfigSection("topics_to_examine", state.configDraft.topics_to_examine, "Detailed topics")
  );
  byId("saveSettingsGeneralFilteringButton").addEventListener("click", () =>
    saveConfigSection("general_filtering", state.configDraft.general_filtering, "General filtering")
  );
  byId("saveSettingsFilteringButton").addEventListener("click", () =>
    saveConfigSection("filtering", state.configDraft.filtering, "Detailed filtering")
  );
  byId("saveSettingsPipelineButton").addEventListener("click", () =>
    saveConfigSection("pipeline", state.configDraft.pipeline, "Pipeline")
  );
  byId("saveLearnedButton").addEventListener("click", saveLearned);
  byId("saveMemoryLearnedButton").addEventListener("click", saveLearned);
  byId("previewLearnedButton").addEventListener("click", () => previewLearned("learnedPreview"));
  byId("previewMemoryLearnedButton").addEventListener("click", () => previewLearned("memoryLearnedPreview"));
  byId("saveStoryIndexButton").addEventListener("click", saveStoryIndex);
  byId("pruneMemoryButton").addEventListener("click", pruneMemory);
  bindMemoryRepairEvents();
  bindRunEvents();
  byId("runAutoconfigButton").addEventListener("click", runAutoconfig);
  byId("toggleReportContent").addEventListener("click", () => togglePane("content"));
  byId("toggleReportBrowser").addEventListener("click", () => togglePane("browser"));
  byId("reportTypeFilter").addEventListener("change", (event) => {
    state.reportType = event.target.value;
    renderReportList();
  });
  byId("reportSort").addEventListener("change", (event) => {
    state.reportSort = event.target.value;
    renderReportList();
  });
  bindMemoryFilters();
}

function bindMemoryFilters() {
  const bindings = [
    ["memoryStorySearch", "storySearch", "input"],
    ["memoryStatusFilter", "status", "change"],
    ["memoryFeedbackActionFilter", "feedbackAction", "change"],
    ["memoryTopicSourceFilter", "topicSource", "input"],
    ["memoryDateBriefFilter", "dateBrief", "input"],
  ];
  bindings.forEach(([id, key, eventName]) => {
    const node = byId(id);
    node.value = state.memoryFilters[key];
    node.addEventListener(eventName, (event) => {
      state.memoryFilters[key] = event.target.value;
      renderMemory();
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setReportMemoryReload(loadMemory);
  setRunRefreshCallbacks({ reports: loadReports, memory: loadMemory });
  bindEvents();
  togglePaneClasses();
  loadInitial();
});

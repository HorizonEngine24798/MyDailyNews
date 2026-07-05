import { api } from "./api.js";
import { byId, escapeAttr, escapeHtml, setStatus } from "./dom.js";
import { selectReport } from "./reports.js";
import { state } from "./state.js";

const refreshedRunIds = new Set();
const RUN_POLL_INTERVAL_MS = 2000;
const RUNNING_STATUSES = new Set(["running", "canceling"]);
let pollTimer = null;
let refreshReports = async () => {};
let refreshMemory = async () => {};

export function setRunRefreshCallbacks(callbacks) {
  refreshReports = typeof callbacks?.reports === "function" ? callbacks.reports : async () => {};
  refreshMemory = typeof callbacks?.memory === "function" ? callbacks.memory : async () => {};
}

export async function loadRuns() {
  const payload = await api("/api/runs");
  state.runs = payload.runs || [];
  if (!state.currentRun && state.runs.length) {
    state.currentRun = state.runs[0];
  } else if (state.currentRun) {
    const fresh = state.runs.find((run) => run.id === state.currentRun.id);
    if (fresh) {
      state.currentRun = fresh;
    }
  }
  renderRuns();
  scheduleRunPoll();
}

export function bindRunEvents() {
  byId("startRunButton").addEventListener("click", startRun);
  byId("refreshRunsButton").addEventListener("click", loadRuns);
  byId("cancelRunButton").addEventListener("click", cancelCurrentRun);
}

export function renderRuns() {
  renderRunList();
  renderRunDetail();
}

async function startRun() {
  const payload = {
    kind: byId("runKind").value,
    brief: byId("runBrief").value,
    date: byId("runDate").value,
    memory_action: byId("runMemoryAction").value,
    debug: byId("runDebug").value === "true",
  };
  try {
    const result = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.currentRun = result.run;
    await loadRuns();
    setStatus(`Started run: ${result.run.label}`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function cancelCurrentRun() {
  if (!state.currentRun || state.currentRun.status !== "running") {
    return;
  }
  if (!window.confirm(`Cancel ${state.currentRun.label}?`)) {
    return;
  }
  try {
    const result = await api(`/api/runs/${encodeURIComponent(state.currentRun.id)}/cancel`, {
      method: "POST",
      body: JSON.stringify({}),
    });
    state.currentRun = result.run;
    await loadRuns();
    setStatus("Run cancellation requested");
  } catch (error) {
    setStatus(error.message, true);
  }
}

function renderRunList() {
  const target = byId("runList");
  const runs = state.runs || [];
  if (!runs.length) {
    target.innerHTML = `<div class="empty">No runs in this GUI session.</div>`;
    return;
  }
  target.innerHTML = runs
    .map(
      (run) => `
        <button class="run-row ${state.currentRun?.id === run.id ? "active" : ""}" type="button" data-run-id="${escapeAttr(run.id)}">
          <strong>${escapeHtml(run.label || run.kind)}</strong>
          <span>${escapeHtml(run.status || "")}</span>
          <span class="muted small">${escapeHtml(run.started_at || "")}</span>
        </button>
      `
    )
    .join("");
  target.querySelectorAll("[data-run-id]").forEach((button) => {
    button.addEventListener("click", () => selectRun(button.dataset.runId));
  });
}

async function selectRun(runId) {
  try {
    const payload = await api(`/api/runs/${encodeURIComponent(runId)}`);
    state.currentRun = payload.run;
    renderRuns();
    scheduleRunPoll();
  } catch (error) {
    setStatus(error.message, true);
  }
}

function renderRunDetail() {
  const run = state.currentRun;
  const detail = byId("runDetail");
  if (!run) {
    detail.innerHTML = `<div class="empty">Select or start a run.</div>`;
    byId("cancelRunButton").disabled = true;
    return;
  }
  byId("cancelRunButton").disabled = run.status !== "running";
  const outputs = run.output_paths || [];
  detail.innerHTML = `
    <div class="run-meta">
      <div class="info-row"><span class="muted small">Status</span><strong>${escapeHtml(run.status || "")}</strong></div>
      <div class="info-row"><span class="muted small">Return code</span><strong>${escapeHtml(run.returncode ?? "")}</strong></div>
      <div class="info-row"><span class="muted small">Started</span><strong>${escapeHtml(run.started_at || "")}</strong></div>
      <div class="info-row"><span class="muted small">Finished</span><strong>${escapeHtml(run.finished_at || "")}</strong></div>
    </div>
    <div class="run-command">${escapeHtml(run.command_display || "")}</div>
    ${run.error ? `<div class="warning-row">${escapeHtml(run.error)}</div>` : ""}
    <h3>Output Files</h3>
    <div class="run-output-list">
      ${
        outputs.length
          ? outputs.map((path) => outputPathRow(path)).join("")
          : `<div class="muted small">No new output files detected yet.</div>`
      }
    </div>
    <h3>Stdout</h3>
    <pre class="console-output run-console">${escapeHtml(run.stdout_tail || "")}</pre>
    <h3>Stderr</h3>
    <pre class="console-output run-console">${escapeHtml(run.stderr_tail || "")}</pre>
  `;
  detail.querySelectorAll("[data-report-id]").forEach((button) => {
    button.addEventListener("click", () => selectReport(button.dataset.reportId));
  });
}

function outputPathRow(path) {
  const reportId = reportIdFromPath(path);
  return `
    <div class="output-path-row">
      <code>${escapeHtml(path)}</code>
      ${reportId ? `<button class="mini-button" type="button" data-report-id="${escapeAttr(reportId)}">Open</button>` : ""}
    </div>
  `;
}

function reportIdFromPath(path) {
  const name = String(path || "").split(/[\\/]/).pop() || "";
  return /^\d{4}-\d{2}-\d{2}_.+\.md$/.test(name) ? name : "";
}

function scheduleRunPoll() {
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
  const shouldPoll = (state.runs || []).some((run) => RUNNING_STATUSES.has(run.status));
  if (!shouldPoll) {
    return;
  }
  pollTimer = window.setTimeout(pollRuns, RUN_POLL_INTERVAL_MS);
}

async function pollRuns() {
  try {
    await loadRuns();
    await refreshAfterCompletedRuns();
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function refreshAfterCompletedRuns() {
  const completed = (state.runs || []).filter((run) => run.status === "completed" && !refreshedRunIds.has(run.id));
  if (!completed.length) {
    return;
  }
  completed.forEach((run) => refreshedRunIds.add(run.id));
  await Promise.all([refreshReports(), refreshMemory()]);
  renderRuns();
}

import { api } from "./api.js";
import { audioButtonLabel, toggleAudio } from "./audio.js";
import { byId, escapeAttr, escapeHtml, ordinal, renderMarkdown, reportDisplayTitle, reportListTitle, setStatus } from "./dom.js";
import { feedbackLabels, monthNames, state } from "./state.js";

let reloadMemory = async () => {};
const MISSING_DATE_SORT_KEY = "9999-99-99";
const UNDATED_GROUP_DATE = "undated-00-00";

export function setReportMemoryReload(callback) {
  reloadMemory = typeof callback === "function" ? callback : async () => {};
}

export function updateReportTypeOptions() {
  const select = byId("reportTypeFilter");
  const previous = select.value || state.reportType || "all";
  const kinds = [...new Set(state.reports.map((report) => report.kind).filter(Boolean))].sort();
  select.innerHTML = [`<option value="all">All types</option>`]
    .concat(kinds.map((kind) => `<option value="${escapeAttr(kind)}">${escapeHtml(kind)}</option>`))
    .join("");
  select.value = kinds.includes(previous) ? previous : "all";
  state.reportType = select.value;
}

export function sortedReports(reports) {
  const rows = [...reports];
  rows.sort((a, b) => {
    if (state.reportSort === "date_desc") {
      return (
        compareReportDate(b, a) ||
        String(a.kind).localeCompare(String(b.kind)) ||
        String(a.title).localeCompare(String(b.title))
      );
    }
    if (state.reportSort === "type") {
      return (
        String(a.kind).localeCompare(String(b.kind)) ||
        compareReportDate(a, b) ||
        String(a.title).localeCompare(String(b.title))
      );
    }
    if (state.reportSort === "title") {
      return String(a.title).localeCompare(String(b.title)) || compareReportDate(a, b);
    }
    return (
      compareReportDate(a, b) ||
      String(a.kind).localeCompare(String(b.kind)) ||
      String(a.title).localeCompare(String(b.title))
    );
  });
  return rows;
}

export function renderReportList() {
  const list = byId("reportList");
  const reports = filteredReports();
  if (!reports.length) {
    list.innerHTML = `<div class="empty">No reports match this filter.</div>`;
    return;
  }

  list.innerHTML = "";
  groupReportsByDate(reports).forEach((yearGroup) => {
    const yearKey = `year:${yearGroup.year}`;
    const yearNode = dateGroupDetails("report-year", yearKey, yearGroup.year, true);

    yearGroup.months.forEach((monthGroup) => {
      const monthKey = `${yearKey}:month:${monthGroup.month}`;
      const monthNode = dateGroupDetails("report-month", monthKey, monthGroup.month, true);

      monthGroup.days.forEach((dayGroup) => {
        const dayKey = `${monthKey}:day:${dayGroup.day}`;
        const dayNode = dateGroupDetails("report-day", dayKey, ordinal(dayGroup.day), true);
        dayGroup.reports.forEach((report) => dayNode.appendChild(reportButton(report)));
        monthNode.appendChild(dayNode);
      });

      yearNode.appendChild(monthNode);
    });

    list.appendChild(yearNode);
  });
}

export async function selectReport(reportId) {
  try {
    state.currentReport = await api(`/api/reports/${encodeURIComponent(reportId)}`);
    renderReportList();
    renderReportDetail();
    setStatus("Ready");
  } catch (error) {
    setStatus(error.message, true);
  }
}

export function renderReportDetail() {
  const report = state.currentReport;
  if (!report) {
    return;
  }
  byId("reportHeader").innerHTML = `
    <div class="report-title-block">
      <h2>${escapeHtml(reportDisplayTitle(report))}</h2>
      <div class="muted small">${escapeHtml(report.date || "undated")} / ${escapeHtml(report.kind || "")}</div>
    </div>
    ${
      report.audio_url
        ? `<button id="playReportAudio" class="command audio-button" type="button" data-audio-url="${escapeAttr(
            report.audio_url
          )}">${escapeHtml(audioButtonLabel(report.audio_url))}</button>`
        : ""
    }
  `;
  const playButton = byId("playReportAudio");
  if (playButton) {
    playButton.addEventListener("click", () => toggleAudio(playButton.dataset.audioUrl, reportDisplayTitle(report)));
  }
  renderFeedback(report.feedback_items || []);
  const cards = Array.isArray(report.json && report.json.claim_context_cards) ? report.json.claim_context_cards : [];
  const markdownView = byId("markdownView");
  markdownView.innerHTML = renderMarkdown(report.markdown || "", cards);
  markdownView.querySelectorAll("[data-claim-ref]").forEach((button) => {
    button.addEventListener("click", () => showClaimContext(cards.find((card) => Number(card.ref) === Number(button.dataset.claimRef))));
  });
}

function showClaimContext(card) {
  if (!card) {
    return;
  }
  let dialog = byId("claimContextDialog");
  if (!dialog) {
    dialog = document.createElement("dialog");
    dialog.id = "claimContextDialog";
    dialog.className = "claim-context-dialog";
    document.body.appendChild(dialog);
  }
  dialog.replaceChildren();
  const title = document.createElement("h2");
  title.textContent = card.title || card.claim || "Claim context";
  dialog.appendChild(title);
  [
    ["Who says it", card.who_says],
    ["What reporting adds", card.reporting_summary],
    ["Evidence check", card.evidence_check],
    ["Verification scope", [card.verification_verdict, card.verdict_scope].filter(Boolean).join(" / ")],
    ["Qualification", card.qualification],
    ["What remains uncertain", card.limitations],
  ].forEach(([label, value]) => {
    if (!value) return;
    const paragraph = document.createElement("p");
    const strong = document.createElement("strong");
    strong.textContent = `${label}: `;
    paragraph.append(strong, document.createTextNode(String(value)));
    dialog.appendChild(paragraph);
  });
  const validSources = (Array.isArray(card.sources) ? card.sources : []).filter((source) => safeHttpUrl(source.url));
  if (validSources.length) {
    const heading = document.createElement("h3");
    heading.textContent = "Sources";
    const list = document.createElement("ul");
    validSources.forEach((source) => {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = safeHttpUrl(source.url);
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = [source.outlet, source.headline].filter(Boolean).join(" — ") || "Source";
      item.appendChild(link);
      list.appendChild(item);
    });
    dialog.append(heading, list);
  }
  const close = document.createElement("button");
  close.type = "button";
  close.className = "command";
  close.textContent = "Close";
  close.addEventListener("click", () => dialog.close());
  dialog.appendChild(close);
  dialog.showModal();
}

function safeHttpUrl(value) {
  try {
    const url = new URL(String(value || ""));
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch (_error) {
    return "";
  }
}

function filteredReports() {
  const reports =
    state.reportType === "all" ? [...state.reports] : state.reports.filter((report) => report.kind === state.reportType);
  return sortedReports(reports);
}

function compareReportDate(a, b) {
  const dateCompare = String(a.date || MISSING_DATE_SORT_KEY).localeCompare(String(b.date || MISSING_DATE_SORT_KEY));
  if (dateCompare !== 0) {
    return dateCompare;
  }
  return String(a.filename).localeCompare(String(b.filename));
}

function dateGroupDetails(className, key, label, defaultOpen) {
  const node = document.createElement("details");
  node.className = className;
  node.dataset.groupKey = key;
  node.open = Object.prototype.hasOwnProperty.call(state.reportGroupsOpen, key) ? state.reportGroupsOpen[key] : defaultOpen;

  const summary = document.createElement("summary");
  summary.className = "date-summary";
  summary.textContent = label;
  node.appendChild(summary);

  node.addEventListener("toggle", (event) => {
    if (event.target === node) {
      state.reportGroupsOpen[key] = node.open;
    }
  });
  return node;
}

function groupReportsByDate(reports) {
  const yearMap = new Map();
  reports.forEach((report) => {
    const parts = String(report.date || UNDATED_GROUP_DATE).split("-");
    const year = parts[0] || "Undated";
    const month = monthNames[Number(parts[1] || 0) - 1] || "Undated";
    const day = Number(parts[2] || 0) || 0;

    if (!yearMap.has(year)) {
      yearMap.set(year, new Map());
    }
    const monthMap = yearMap.get(year);
    if (!monthMap.has(month)) {
      monthMap.set(month, new Map());
    }
    const dayMap = monthMap.get(month);
    if (!dayMap.has(day)) {
      dayMap.set(day, []);
    }
    dayMap.get(day).push(report);
  });

  return [...yearMap.entries()].map(([year, monthMap]) => ({
    year,
    months: [...monthMap.entries()].map(([month, dayMap]) => ({
      month,
      days: [...dayMap.entries()].map(([day, dayReports]) => ({ day, reports: dayReports })),
    })),
  }));
}

function reportButton(report) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "report-row";
  button.classList.toggle("active", state.currentReport && state.currentReport.id === report.id);
  button.innerHTML = `<strong>${escapeHtml(reportListTitle(report))}</strong>`;
  button.addEventListener("click", () => selectReport(report.id));
  return button;
}

function renderFeedback(items) {
  const panel = byId("feedbackPanel");
  if (!items.length) {
    panel.innerHTML = `<div class="muted small">No feedback targets.</div>`;
    return;
  }

  const actions = (state.app && state.app.feedback_actions) || Object.keys(feedbackLabels);
  panel.innerHTML = "";
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "feedback-row";

    const title = document.createElement("div");
    const latestAction = item.latest_feedback_action || "";
    const latestRowId =
      Array.isArray(item.feedback_history) && item.feedback_history.length ? item.feedback_history[0].row_id : "";
    const badges = [];
    if (item.score !== undefined && item.score !== null) {
      const rankNote =
        item.selection_rank_score !== undefined && Math.abs(Number(item.selection_rank_score) - Number(item.score)) > 0.01
          ? ` -> ${Number(item.selection_rank_score).toFixed(2)}`
          : "";
      badges.push(`<span class="taste-badge neutral">Score ${escapeHtml(Number(item.score).toFixed(2))}${escapeHtml(rankNote)}</span>`);
    }
    if (item.learned_preference_adjustment) {
      const adj = Number(item.learned_preference_adjustment);
      const tone = adj >= 0 ? "positive" : "negative";
      const sign = adj >= 0 ? "+" : "";
      const why = item.learned_preference_reason ? ` (${escapeHtml(item.learned_preference_reason)})` : "";
      badges.push(`<span class="taste-badge ${tone}">Preference ${sign}${adj.toFixed(2)}${why}</span>`);
    }
    title.innerHTML = `
      <strong>${escapeHtml(item.title || item.id)}</strong>
      <div class="muted small">${escapeHtml(item.source || "")}${item.topic ? " / " + escapeHtml(item.topic) : ""}</div>
      ${badges.length ? `<div class="taste-badge-row">${badges.join("")}</div>` : ""}
      ${
        item.feedback_count
          ? `<div class="muted small">Feedback recorded: ${escapeHtml(feedbackLabels[latestAction] || latestAction)}${
              item.feedback_count > 1 ? ` (${escapeHtml(item.feedback_count)} total)` : ""
            }</div>`
          : ""
      }
    `;

    const buttons = document.createElement("div");
    buttons.className = "feedback-actions";
    actions.forEach((action) => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = feedbackLabels[action] || action;
      button.classList.toggle("active", action === latestAction);
      button.addEventListener("click", () => sendFeedback(action, item));
      buttons.appendChild(button);
    });
    if (latestRowId) {
      const undoButton = document.createElement("button");
      undoButton.type = "button";
      undoButton.className = "mini-button";
      undoButton.textContent = "Undo";
      undoButton.title = "Remove latest feedback event on this article";
      undoButton.addEventListener("click", () => undoReportFeedback(latestRowId));
      buttons.appendChild(undoButton);
    }

    row.appendChild(title);
    row.appendChild(buttons);
    panel.appendChild(row);
  });
}

async function undoReportFeedback(rowId) {
  if (!state.currentReport || !rowId) {
    return;
  }
  try {
    await api("/api/memory/repair", {
      method: "POST",
      body: JSON.stringify({ action: "feedback_delete", row_ids: [rowId], confirm: true }),
    });
    state.currentReport = await api(`/api/reports/${encodeURIComponent(state.currentReport.id)}`);
    renderReportList();
    renderReportDetail();
    await reloadMemory();
    setStatus("Removed feedback event.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function sendFeedback(action, item) {
  if (!state.currentReport) {
    return;
  }

  try {
    const reportId = state.currentReport.id;
    const result = await api("/api/feedback", {
      method: "POST",
      body: JSON.stringify({
        action,
        report_id: reportId,
        brief_name: state.currentReport.brief_name,
        report_date: state.currentReport.date,
        item,
      }),
    });
    state.currentReport = await api(`/api/reports/${encodeURIComponent(reportId)}`);
    renderReportList();
    renderReportDetail();
    await reloadMemory();
    let learnedStatus = "; no learned preference change";
    if (result.learned_preferences_changed) {
      learnedStatus = "; learned preferences updated";
    } else if (result.story_cooldown && result.story_cooldown.active) {
      learnedStatus = `; story cooldown active (${Number(result.story_cooldown.days_remaining || 7).toFixed(1)}d)`;
    }
    setStatus(`Recorded feedback: ${feedbackLabels[action] || action}${learnedStatus}`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

import { api, clone } from "./api.js";
import { byId, escapeHtml, setStatus } from "./dom.js";
import { renderForm } from "./forms.js";
import { defaultStoryIndex, state } from "./state.js";

export function renderMemory() {
  if (!state.memory) {
    return;
  }

  const summary = state.memory.summary || {};
  const cells = [
    ["Coverage", summary.coverage_records],
    ["Stories", summary.story_index_records],
    ["Active", summary.story_index_active],
    ["Stale", summary.story_index_stale],
    ["Feedback", summary.feedback_events],
    ["Warnings", summary.health_warnings || 0],
    ["Enabled", summary.memory_enabled ? "Yes" : "No"],
  ];

  byId("memorySummary").innerHTML = cells
    .map(([label, value]) => `<div class="summary-cell"><span class="muted small">${label}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");

  renderForm("storyIndexForm", () => state.storyIndexDraft);
  renderTable(
    "storyTable",
    filterStories(state.memory.story_index || []),
    [
      ["title", "Title"],
      ["topic", "Topic"],
      ["family", "Family"],
      ["coverage_count", "Coverage"],
      ["first_seen", "First seen"],
      ["last_seen", "Last seen"],
      ["status", "Status"],
    ],
    [
      { label: "Delete", action: "story-delete" },
    ]
  );

  const coverage = filterCoverage(state.memory.coverage_records || []).sort((a, b) => String(b.date).localeCompare(String(a.date)));
  renderTable(
    "coverageTable",
    coverage,
    [
      ["date", "Date"],
      ["brief_name", "Brief"],
      ["title", "Title"],
      ["prominence", "Prominence"],
      ["story_key", "Story key"],
      ["family", "Family"],
      ["row_id", "Row ID"],
    ],
    [
      { label: "Split", action: "coverage-split" },
      { label: "Archive", action: "coverage-archive" },
      { label: "Delete", action: "coverage-delete" },
    ]
  );

  const feedback = filterFeedback(state.memory.feedback_events || []).sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
  renderTable(
    "feedbackTable",
    feedback,
    [
      ["created_date", "Created"],
      ["action", "Action"],
      ["title", "Title"],
      ["source", "Source"],
      ["topic", "Topic"],
      ["story_key", "Story key"],
      ["row_id", "Row ID"],
    ],
    [
      { label: "Split", action: "feedback-split" },
      { label: "Edit", action: "feedback-edit" },
      { label: "Delete", action: "feedback-delete" },
    ]
  );

  renderLearnedSummary();
  renderRecallPackets();
  renderHealth();
}

function filterStories(rows) {
  const filters = state.memoryFilters || {};
  return rows.filter((row) => {
    if (filters.status && filters.status !== "all" && String(row.status || "") !== filters.status) {
      return false;
    }
    if (!matchesText(row, filters.storySearch, ["story_key", "story_family_key", "family", "title"])) {
      return false;
    }
    if (!matchesText(row, filters.topicSource, ["topic"])) {
      return false;
    }
    return matchesText(row, filters.dateBrief, ["first_seen", "last_seen"]);
  });
}

function filterCoverage(rows) {
  const filters = state.memoryFilters || {};
  return rows.filter(
    (row) =>
      matchesText(row, filters.storySearch, ["story_key", "story_family_key", "family", "title"]) &&
      matchesText(row, filters.topicSource, ["title", "family"]) &&
      matchesText(row, filters.dateBrief, ["date", "brief_name"])
  );
}

function filterFeedback(rows) {
  const filters = state.memoryFilters || {};
  return rows.filter((row) => {
    if (filters.feedbackAction && filters.feedbackAction !== "all" && String(row.action || "") !== filters.feedbackAction) {
      return false;
    }
    return (
      matchesText(row, filters.storySearch, ["story_key", "story_family_key", "title", "article_id"]) &&
      matchesText(row, filters.topicSource, ["topic", "source"]) &&
      matchesText(row, filters.dateBrief, ["created_date", "created_at", "report_date", "brief_name"])
    );
  });
}

function matchesText(row, needle, keys) {
  const query = normalize(needle);
  if (!query) {
    return true;
  }
  return keys.some((key) => normalize(row[key]).includes(query));
}

function normalize(value) {
  return String(value == null ? "" : value).trim().toLowerCase();
}

export function bindMemoryRepairEvents() {
  byId("repairMergeButton").addEventListener("click", repairMergeStories);
  byId("repairSplitButton").addEventListener("click", repairSplitStory);
}

function renderTable(id, rows, columns, actions = []) {
  const target = byId(id);
  if (!rows.length) {
    target.innerHTML = `<div class="empty">No rows.</div>`;
    return;
  }

  const head = columns.map(([, label]) => `<th>${escapeHtml(label)}</th>`).join("");
  const actionHead = actions.length ? `<th>Actions</th>` : "";
  const body = rows
    .map((row, rowIndex) => {
      const cells = columns.map(([key]) => `<td>${escapeHtml(row[key] == null ? "" : row[key])}</td>`).join("");
      const actionCells = actions.length
        ? `<td><div class="table-actions">${actions
            .map(
              (action, actionIndex) =>
                `<button class="mini-button" type="button" data-table-action="${escapeHtml(action.action)}" data-row-index="${rowIndex}" data-action-index="${actionIndex}">${escapeHtml(action.label)}</button>`
            )
            .join("")}</div></td>`
        : "";
      return `<tr>${cells}${actionCells}</tr>`;
    })
    .join("");
  target.innerHTML = `<table><thead><tr>${head}${actionHead}</tr></thead><tbody>${body}</tbody></table>`;
  target.querySelectorAll("[data-table-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const row = rows[Number(button.dataset.rowIndex || 0)];
      const action = actions[Number(button.dataset.actionIndex || 0)];
      handleTableAction(action.action, row);
    });
  });
}

function handleTableAction(action, row) {
  if (action === "story-delete") {
    repairMemory(
      {
        action: "story_delete",
        story_key: row.story_key,
      },
      `Delete story ${row.story_key}`
    );
  } else if (action === "coverage-archive" || action === "coverage-delete") {
    repairMemory(
      {
        action: action === "coverage-archive" ? "coverage_archive" : "coverage_delete",
        row_ids: [row.row_id],
      },
      `${action === "coverage-archive" ? "Archive" : "Delete"} coverage row`
    );
  } else if (action === "coverage-split") {
    byId("repairSplitSource").value = row.story_key || byId("repairSplitSource").value;
    appendTextareaValue("repairSplitCoverageRows", row.row_id);
    setStatus("Coverage row added to split draft");
  } else if (action === "feedback-split") {
    byId("repairSplitSource").value = row.story_key || byId("repairSplitSource").value;
    appendTextareaValue("repairSplitFeedbackRows", row.row_id);
    setStatus("Feedback row added to split draft");
  } else if (action === "feedback-delete") {
    repairMemory(
      {
        action: "feedback_delete",
        row_ids: [row.row_id],
      },
      "Delete feedback event"
    );
  } else if (action === "feedback-edit") {
    editFeedbackEvent(row);
  }
}

async function repairMemory(payload, label) {
  if (!window.confirm(`${label}? A backup will be created first.`)) {
    return;
  }
  try {
    const result = await api("/api/memory/repair", {
      method: "POST",
      body: JSON.stringify({ ...payload, confirm: true }),
    });
    state.memory = result.memory;
    state.storyIndexDraft = clone(state.memory.story_index_file || defaultStoryIndex());
    renderMemory();
    const backup = result.repair?.backup?.path || "";
    setStatus(backup ? `${label} complete; backup ${backup}` : `${label} complete`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

function editFeedbackEvent(row) {
  const draft = {
    action: row.action || "",
    report_date: row.report_date || "",
    brief_name: row.brief_name || "",
    article_id: row.article_id || "",
    story_key: row.story_key || "",
    story_family_key: row.story_family_key || "",
    title: row.title || "",
    source: row.source || "",
    topic: row.topic || "",
    notes: row.notes || "",
  };
  const raw = window.prompt("Feedback event JSON", JSON.stringify(draft, null, 2));
  if (raw == null) {
    return;
  }
  let event;
  try {
    event = JSON.parse(raw);
  } catch (error) {
    setStatus(`Invalid feedback JSON: ${error.message}`, true);
    return;
  }
  repairMemory(
    {
      action: "feedback_edit",
      row_ids: [row.row_id],
      event,
    },
    "Edit feedback event"
  );
}

function repairMergeStories() {
  const sourceStoryKeys = lineValues(byId("repairMergeSources").value);
  let canonicalStory;
  try {
    canonicalStory = parseJsonObject(byId("repairMergeCanonical").value, "Canonical story JSON");
  } catch (error) {
    setStatus(error.message, true);
    return;
  }
  repairMemory(
    {
      action: "story_merge",
      source_story_keys: sourceStoryKeys,
      canonical_story: canonicalStory,
    },
    "Merge stories"
  );
}

function repairSplitStory() {
  let newStory;
  try {
    newStory = parseJsonObject(byId("repairSplitNewStory").value, "New story JSON");
  } catch (error) {
    setStatus(error.message, true);
    return;
  }
  repairMemory(
    {
      action: "story_split",
      source_story_key: byId("repairSplitSource").value,
      coverage_row_ids: lineValues(byId("repairSplitCoverageRows").value),
      feedback_row_ids: lineValues(byId("repairSplitFeedbackRows").value),
      new_story: newStory,
    },
    "Split story"
  );
}

function parseJsonObject(raw, label) {
  const text = String(raw || "").trim();
  if (!text) {
    return {};
  }
  const value = JSON.parse(text);
  if (!value || Array.isArray(value) || typeof value !== "object") {
    throw new Error(`${label} must be a JSON object`);
  }
  return value;
}

function lineValues(value) {
  return String(value || "")
    .split(/\r?\n|,/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function appendTextareaValue(id, value) {
  const node = byId(id);
  const values = lineValues(node.value);
  if (!values.includes(value)) {
    values.push(value);
  }
  node.value = values.join("\n");
}

function renderLearnedSummary() {
  const summary = state.memory.learned_preferences_summary || {};
  renderInfoList("learnedMemorySummary", [
    ["File", summary.exists ? "Present" : "Not created"],
    ["Updated", summary.updated_at || ""],
    ["Preferred topics", summary.preferred_topics || 0],
    ["Avoided topics", summary.avoided_topics || 0],
    ["Preferred sources", summary.preferred_sources || 0],
    ["Avoided sources", summary.avoided_sources || 0],
    ["Topic weights", summary.topic_weights || 0],
    ["Source weights", summary.source_weights || 0],
    ["Notes", summary.has_notes ? "Present" : "Empty"],
  ]);
}

function renderRecallPackets() {
  const packets = state.memory.recall_packets || {};
  const latest = packets.latest || {};
  renderInfoList("recallPacketSummary", [
    ["Status", packets.exists ? "Present" : "None"],
    ["Count", packets.count || 0],
    ["Latest date", latest.date || ""],
    ["Latest brief", latest.brief_name || ""],
    ["Path", latest.path || ""],
  ]);
}

function renderHealth() {
  const health = state.memory.health || {};
  const warnings = health.warnings || [];
  const target = byId("memoryHealth");
  if (!warnings.length) {
    target.innerHTML = `<div class="empty">No warnings.</div>`;
    return;
  }
  target.innerHTML = warnings
    .map((warning) => {
      const details = []
        .concat(warning.story_keys || [])
        .concat(warning.line_numbers || [])
        .concat((warning.events || []).map((event) => event.title || event.action || event.created_at || "event"))
        .join(", ");
      return `
        <div class="info-row warning-row">
          <span>${escapeHtml(warning.message || warning.code || "Warning")}</span>
          <strong>${escapeHtml(warning.count || "")}</strong>
          ${details ? `<div class="muted small">${escapeHtml(details)}</div>` : ""}
        </div>
      `;
    })
    .join("");
}

function renderInfoList(id, rows) {
  const target = byId(id);
  target.innerHTML = rows
    .map(
      ([label, value]) => `
        <div class="info-row">
          <span class="muted small">${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `
    )
    .join("");
}

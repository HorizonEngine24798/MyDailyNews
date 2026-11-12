import { api } from "./api.js";
import { byId, escapeAttr, escapeHtml, setStatus } from "./dom.js";
import { feedbackLabels, state } from "./state.js";

const tasteUi = {
  filter: "all",
  query: "",
  mergeSelected: new Set(),
  mergeQuery: "",
  splitSourceKey: "",
  splitCoverageIds: new Set(),
  splitFeedbackIds: new Set(),
};

let refreshMemoryCallback = async () => {};

export function setTasteRefreshCallback(callback) {
  refreshMemoryCallback = typeof callback === "function" ? callback : async () => {};
}

export function renderTasteDashboard(container) {
  if (!container) {
    return;
  }
  const memory = state.memory || {};
  const taste = memory.taste_summary || {};
  const counts = taste.feedback_counts || {};
  const semantic = taste.semantic_engine || {};
  const activeCooldowns = Array.isArray(taste.active_cooldowns) ? taste.active_cooldowns : [];
  const feedbackEvents = Array.isArray(taste.recent_feedback)
    ? taste.recent_feedback
    : Array.isArray(memory.feedback_events)
    ? [...memory.feedback_events].sort((a, b) => String(b.created_at || "").localeCompare(String(a.created_at || "")))
    : [];

  const totalFeedback = Number(taste.total_feedback ?? feedbackEvents.length ?? 0);
  const likedCount = Number(taste.liked_count ?? counts.more_like_this ?? 0);
  const dislikedCount = Number(
    taste.disliked_count ?? Number(counts.not_relevant || 0) + Number(counts.not_interested_in_topic || 0)
  );
  const repetitiveCount = Number(taste.repetitive_count ?? counts.too_repetitive ?? 0);
  const learnedPrefs = ensureLearnedDraft();

  container.innerHTML = `
    <div class="taste-dashboard">
      <div class="taste-card">
        <div class="taste-card-header">
          <div>
            <h3>Taste &amp; Semantic Memory</h3>
            <div class="muted small">
              Engine: ${escapeHtml(semantic.model || "intfloat/multilingual-e5-small")}
              (${semantic.available ? "Active on CPU" : "Lexical fallback mode"}) &middot;
              Decay half-life: ${escapeHtml(String(semantic.half_life_days || 30))} days
            </div>
          </div>
          <div class="button-row">
            <button id="tasteRebuildDecay" class="command" type="button">Rebuild Weights (30d Decay)</button>
            <button id="tasteResetWeights" class="command" type="button">Reset Learned Weights</button>
          </div>
        </div>
        <div class="taste-stats-grid">
          <div class="taste-stat-pill">
            <span class="muted small">Liked summaries</span>
            <strong>${likedCount}</strong>
            <span class="small muted">More like this</span>
          </div>
          <div class="taste-stat-pill">
            <span class="muted small">Disliked summaries</span>
            <strong>${dislikedCount}</strong>
            <span class="small muted">Not relevant / topic</span>
          </div>
          <div class="taste-stat-pill">
            <span class="muted small">Repetitive flags</span>
            <strong>${repetitiveCount}</strong>
            <span class="small muted">${activeCooldowns.length} active cooldowns</span>
          </div>
          <div class="taste-stat-pill">
            <span class="muted small">Topic weights</span>
            <strong>${Object.keys(learnedPrefs.topic_weights || {}).length}</strong>
            <span class="small muted">${Object.keys(learnedPrefs.source_weights || {}).length} source weights</span>
          </div>
          <div class="taste-stat-pill">
            <span class="muted small">Total rated items</span>
            <strong>${totalFeedback}</strong>
            <span class="small muted">Across reports</span>
          </div>
        </div>
      </div>

      <div class="taste-split-grid">
        <div class="taste-card">
          <div class="taste-card-header">
            <div>
              <h3>Rated News Summaries</h3>
              <div class="muted small">Review, flip, or remove feedback you recorded on articles.</div>
            </div>
          </div>
          <div class="taste-filter-pills">
            ${filterPill("all", `All (${feedbackEvents.length})`)}
            ${filterPill("liked", `Liked (${likedCount})`)}
            ${filterPill("disliked", `Disliked (${dislikedCount})`)}
            ${filterPill("repetitive", `Repetitive (${repetitiveCount})`)}
            <input id="tasteFeedSearch" class="field-input" type="search" value="${escapeAttr(tasteUi.query)}" placeholder="Search title, source, or topic..." style="max-width: 15rem; margin-left: auto;" />
          </div>
          <div id="tasteFeedbackFeed" class="taste-feed">
            ${renderFeedbackFeed(feedbackEvents)}
          </div>
        </div>

        <div class="taste-card">
          <div class="taste-card-header">
            <div>
              <h3>Active Story Cooldowns</h3>
              <div class="muted small">Stories suppressed after "Too repetitive" feedback (auto-decays over 7 days).</div>
            </div>
          </div>
          <div id="tasteCooldownList" class="taste-feed">
            ${renderCooldownList(activeCooldowns, feedbackEvents)}
          </div>
        </div>
      </div>

      <div class="taste-card">
        <div class="taste-card-header">
          <div>
            <h3>Visual Preference Editor</h3>
            <div class="muted small">Manage preferred/avoided topic &amp; source chips and tune semantic preference weights without JSON.</div>
          </div>
          <div class="button-row">
            <button id="tasteSaveVisualPrefs" class="command primary" type="button">Save Preferences</button>
          </div>
        </div>
        <div class="taste-visual-repair-grid">
          ${renderChipEditor("preferred_topics", "Preferred Topics", "positive", learnedPrefs.preferred_topics)}
          ${renderChipEditor("avoided_topics", "Avoided Topics", "negative", learnedPrefs.avoided_topics)}
          ${renderChipEditor("preferred_sources", "Preferred Sources", "positive", learnedPrefs.preferred_sources)}
          ${renderChipEditor("avoided_sources", "Avoided Sources", "negative", learnedPrefs.avoided_sources)}
        </div>
        <div class="taste-visual-repair-grid">
          ${renderWeightEditor("topic_weights", "Topic Weights (-3 to +3)", learnedPrefs.topic_weights)}
          ${renderWeightEditor("source_weights", "Source Weights (-3 to +3)", learnedPrefs.source_weights)}
        </div>
        <div class="field-row">
          <label class="field-label" for="tasteVisualNotes">Notes</label>
          <input id="tasteVisualNotes" class="field-input" type="text" value="${escapeAttr(learnedPrefs.notes || "")}" placeholder="Optional note about your evolving reading preferences..." />
        </div>
      </div>

      <div class="taste-card">
        <div class="taste-card-header">
          <div>
            <h3>Visual Story Merge &amp; Split</h3>
            <div class="muted small">Combine duplicate stories or split miscategorized coverage without editing raw JSON.</div>
          </div>
        </div>
        <div class="taste-visual-repair-grid">
          ${renderVisualMerge(memory.story_store || [])}
          ${renderVisualSplit(memory.story_store || [], memory.coverage_records || [], memory.feedback_events || [])}
        </div>
      </div>
    </div>
  `;

  bindTasteEvents(container);
}

function ensureLearnedDraft() {
  if (!state.learnedDraft || typeof state.learnedDraft !== "object") {
    const raw = (state.learned && state.learned.preferences) || {};
    state.learnedDraft = {
      schema_version: Number(raw.schema_version || 1),
      updated_at: String(raw.updated_at || ""),
      preferred_topics: [...(raw.preferred_topics || [])],
      avoided_topics: [...(raw.avoided_topics || [])],
      preferred_sources: [...(raw.preferred_sources || [])],
      avoided_sources: [...(raw.avoided_sources || [])],
      topic_weights: { ...(raw.topic_weights || {}) },
      source_weights: { ...(raw.source_weights || {}) },
      notes: String(raw.notes || ""),
    };
  }
  return state.learnedDraft;
}

function filterPill(id, label) {
  const active = tasteUi.filter === id ? "active" : "";
  return `<button class="taste-pill-btn ${active}" type="button" data-taste-filter="${escapeAttr(id)}">${escapeHtml(label)}</button>`;
}

function filteredFeedbackEvents(events) {
  const query = tasteUi.query.trim().toLowerCase();
  return events.filter((event) => {
    const action = String(event.action || "");
    if (tasteUi.filter === "liked" && action !== "more_like_this") {
      return false;
    }
    if (tasteUi.filter === "disliked" && !["not_relevant", "not_interested_in_topic"].includes(action)) {
      return false;
    }
    if (tasteUi.filter === "repetitive" && action !== "too_repetitive") {
      return false;
    }
    if (!query) {
      return true;
    }
    const haystack = [event.title, event.source, event.topic, event.report_date, event.brief_name]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(query);
  });
}

function renderFeedbackFeed(events) {
  const rows = filteredFeedbackEvents(events);
  if (!rows.length) {
    return `<div class="empty">No recorded article feedback matches the current filter yet. Rate headlines in Reports to build your taste profile.</div>`;
  }
  return rows
    .slice(0, 60)
    .map((event) => {
      const action = String(event.action || "");
      const badgeTone =
        action === "more_like_this" ? "positive" : action === "too_repetitive" ? "warn" : "negative";
      const decayPct =
        event.decay_weight !== undefined && event.decay_weight !== null
          ? `${Math.round(Number(event.decay_weight) * 100)}% decay wt`
          : "";
      const rowId = String(event.row_id || "");
      const meta = [event.source, event.topic, event.report_date || event.created_date, event.brief_name]
        .filter(Boolean)
        .join(" / ");
      return `
        <div class="taste-event-card" data-row-id="${escapeAttr(rowId)}">
          <div class="taste-event-top">
            <div>
              <strong>${escapeHtml(event.title || event.story_key || "Untitled article")}</strong>
              <div class="muted small">${escapeHtml(meta || "No metadata")}</div>
            </div>
            <span class="taste-badge ${badgeTone}">${escapeHtml(feedbackLabels[action] || action)}</span>
          </div>
          <div class="taste-event-actions">
            ${decayPct ? `<span class="taste-badge">${escapeHtml(decayPct)}</span>` : ""}
            <select class="field-input" style="max-width: 12rem; min-height: 30px; padding: 3px 8px;" data-taste-action-select="${escapeAttr(rowId)}">
              ${Object.entries(feedbackLabels)
                .map(
                  ([key, label]) =>
                    `<option value="${escapeAttr(key)}" ${key === action ? "selected" : ""}>${escapeHtml(label)}</option>`
                )
                .join("")}
            </select>
            <button class="mini-button" type="button" data-taste-edit-row="${escapeAttr(rowId)}">Switch</button>
            <button class="mini-button" type="button" data-taste-delete-row="${escapeAttr(rowId)}">Undo</button>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderCooldownList(cooldowns, feedbackEvents) {
  if (!cooldowns.length) {
    return `<div class="empty">No active story cooldowns. Flagging an item as "Too repetitive" suppresses repeat headlines for up to 7 days.</div>`;
  }
  return cooldowns
    .map((entry) => {
      const matchingRows = feedbackEvents
        .filter(
          (row) =>
            String(row.action || "") === "too_repetitive" &&
            ((entry.story_key && row.story_key === entry.story_key) ||
              (entry.story_family_key && row.story_family_key === entry.story_family_key))
        )
        .map((row) => row.row_id)
        .filter(Boolean);
      return `
        <div class="taste-event-card">
          <div class="taste-event-top">
            <div>
              <strong>${escapeHtml(entry.title || entry.story_key || "Story cooldown")}</strong>
              <div class="muted small">${escapeHtml(entry.story_key || entry.story_family_key || "")}${
                entry.topic ? " / " + escapeHtml(entry.topic) : ""
              }</div>
            </div>
            <span class="taste-badge warn">-${Number(entry.penalty || 0).toFixed(2)} rank</span>
          </div>
          <div class="taste-event-actions">
            <span class="taste-badge">${Number(entry.days_remaining || 0).toFixed(1)}d left</span>
            <span class="taste-badge">${escapeHtml(String(entry.repeat_count || 1))}x flagged</span>
            ${
              matchingRows.length
                ? `<button class="mini-button" type="button" data-taste-lift-cooldown="${escapeAttr(
                    JSON.stringify(matchingRows)
                  )}">Lift Cooldown</button>`
                : ""
            }
          </div>
        </div>
      `;
    })
    .join("");
}

function renderChipEditor(field, label, tone, values) {
  const items = Array.isArray(values) ? values : [];
  return `
    <div class="repair-block">
      <div class="field-label">${escapeHtml(label)} (${items.length})</div>
      <div class="taste-chips">
        ${
          items.length
            ? items
                .map(
                  (item) => `
                  <span class="taste-chip ${tone}">
                    <span>${escapeHtml(item)}</span>
                    <button type="button" aria-label="Remove ${escapeAttr(item)}" data-chip-remove="${escapeAttr(field)}" data-chip-value="${escapeAttr(item)}">&times;</button>
                  </span>
                `
                )
                .join("")
            : `<span class="muted small">None set</span>`
        }
      </div>
      <div class="taste-add-row">
        <input class="field-input" type="text" data-chip-input="${escapeAttr(field)}" placeholder="Add ${escapeAttr(label.toLowerCase())}..." />
        <button class="mini-button" type="button" data-chip-add="${escapeAttr(field)}">+ Add</button>
      </div>
    </div>
  `;
}

function renderWeightEditor(field, label, weightsObj) {
  const entries = Object.entries(weightsObj || {}).sort(
    (a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1])) || a[0].localeCompare(b[0])
  );
  return `
    <div class="repair-block">
      <div class="field-label">${escapeHtml(label)} (${entries.length})</div>
      <div class="taste-feed" style="max-height: 220px;">
        ${
          entries.length
            ? entries
                .map(([name, val]) => {
                  const score = Number(val || 0);
                  const tone = score >= 0 ? "positive" : "negative";
                  const sign = score >= 0 ? "+" : "";
                  return `
                    <div class="taste-weight-row">
                      <strong class="small">${escapeHtml(name)}</strong>
                      <input type="range" min="-3" max="3" step="0.05" value="${escapeAttr(String(score))}" data-weight-slider="${escapeAttr(field)}" data-weight-key="${escapeAttr(name)}" />
                      <span class="taste-badge ${tone}" data-weight-readout="${escapeAttr(field + ":" + name)}">${sign}${score.toFixed(2)}</span>
                      <button class="mini-button" type="button" data-weight-remove="${escapeAttr(field)}" data-weight-key="${escapeAttr(name)}">&times;</button>
                    </div>
                  `;
                })
                .join("")
            : `<div class="muted small">No learned weights yet.</div>`
        }
      </div>
      <div class="taste-add-row">
        <input class="field-input" type="text" data-weight-new-key="${escapeAttr(field)}" placeholder="Add item & weight (+0.5)..." />
        <button class="mini-button" type="button" data-weight-add="${escapeAttr(field)}">+ Add</button>
      </div>
    </div>
  `;
}

function renderVisualMerge(stories) {
  const query = tasteUi.mergeQuery.trim().toLowerCase();
  const visible = stories
    .filter((story) => !query || `${story.title} ${story.story_key} ${story.topic}`.toLowerCase().includes(query))
    .slice(0, 60);
  return `
    <div class="repair-block">
      <strong>Visual Story Merge</strong>
      <span class="muted small">Check 2 or more stories representing the same ongoing event.</span>
      <input id="visualMergeSearch" class="field-input" type="search" value="${escapeAttr(tasteUi.mergeQuery)}" placeholder="Filter stories to merge..." />
      <div class="taste-check-list">
        ${
          visible.length
            ? visible
                .map((story) => {
                  const key = String(story.story_key || "");
                  const checked = tasteUi.mergeSelected.has(key) ? "checked" : "";
                  return `
                    <label class="taste-check-item">
                      <input type="checkbox" data-merge-check="${escapeAttr(key)}" ${checked} />
                      <span>
                        <strong>${escapeHtml(story.title || key)}</strong>
                        <span class="muted small"> (${escapeHtml(key)} &middot; ${escapeHtml(String(story.coverage_count || 0))} cov)</span>
                      </span>
                    </label>
                  `;
                })
                .join("")
            : `<span class="muted small">No stories in Store.</span>`
        }
      </div>
      <input id="visualMergeTitle" class="field-input" type="text" placeholder="Optional canonical title for merged story..." />
      <button id="visualMergeSubmit" class="command" type="button">Merge Selected (${tasteUi.mergeSelected.size})</button>
    </div>
  `;
}

function renderVisualSplit(stories, coverageRecords, feedbackEvents) {
  if (!tasteUi.splitSourceKey && stories.length) {
    const withCoverage = stories.find((story) => Number(story.coverage_count || 0) > 0) || stories[0];
    tasteUi.splitSourceKey = String(withCoverage.story_key || "");
  }
  const sourceKey = tasteUi.splitSourceKey;
  const matchingCoverage = coverageRecords.filter((row) => String(row.story_key || "") === sourceKey);
  const matchingFeedback = feedbackEvents.filter((row) => String(row.story_key || "") === sourceKey);

  return `
    <div class="repair-block">
      <strong>Visual Story Split</strong>
      <span class="muted small">Select a story and pick coverage rows to branch into a new story.</span>
      <select id="visualSplitSelect" class="field-input">
        <option value="">Select source story...</option>
        ${stories
          .slice(0, 120)
          .map(
            (story) =>
              `<option value="${escapeAttr(story.story_key)}" ${
                story.story_key === sourceKey ? "selected" : ""
              }>${escapeHtml(story.title || story.story_key)} (${escapeHtml(String(story.coverage_count || 0))})</option>`
          )
          .join("")}
      </select>
      <div class="taste-check-list">
        ${
          matchingCoverage.length
            ? matchingCoverage
                .map((row) => {
                  const id = String(row.row_id || "");
                  const checked = tasteUi.splitCoverageIds.has(id) ? "checked" : "";
                  return `
                    <label class="taste-check-item">
                      <input type="checkbox" data-split-cov="${escapeAttr(id)}" ${checked} />
                      <span>${escapeHtml(row.date || "")} [${escapeHtml(row.brief_name || "")}] ${escapeHtml(row.title || id)}</span>
                    </label>
                  `;
                })
                .join("")
            : `<span class="muted small">No coverage rows linked to this story.</span>`
        }
        ${matchingFeedback
          .map((row) => {
            const id = String(row.row_id || "");
            const checked = tasteUi.splitFeedbackIds.has(id) ? "checked" : "";
            return `
              <label class="taste-check-item">
                <input type="checkbox" data-split-fb="${escapeAttr(id)}" ${checked} />
                <span>Feedback: ${escapeHtml(feedbackLabels[row.action] || row.action)} — ${escapeHtml(row.title || id)}</span>
              </label>
            `;
          })
          .join("")}
      </div>
      <div class="taste-add-row">
        <input id="visualSplitTitle" class="field-input" type="text" placeholder="New story title..." />
        <input id="visualSplitTopic" class="field-input" type="text" placeholder="Topic (optional)" style="max-width: 9rem;" />
      </div>
      <button id="visualSplitSubmit" class="command" type="button">Split Checked Rows Into New Story</button>
    </div>
  `;
}

function bindTasteEvents(container) {
  byId("tasteRebuildDecay")?.addEventListener("click", () => resetPreferences("rebuild_decayed"));
  byId("tasteResetWeights")?.addEventListener("click", () => resetPreferences("weights"));
  byId("tasteSaveVisualPrefs")?.addEventListener("click", saveVisualPreferences);

  const search = byId("tasteFeedSearch");
  search?.addEventListener("input", () => {
    tasteUi.query = search.value;
    const feed = byId("tasteFeedbackFeed");
    if (feed) {
      const memory = state.memory || {};
      const events = (memory.taste_summary && memory.taste_summary.recent_feedback) || memory.feedback_events || [];
      feed.innerHTML = renderFeedbackFeed(events);
      bindFeedRowActions(feed);
    }
  });

  container.querySelectorAll("[data-taste-filter]").forEach((btn) => {
    btn.addEventListener("click", () => {
      tasteUi.filter = btn.dataset.tasteFilter || "all";
      renderTasteDashboard(container);
    });
  });

  bindFeedRowActions(container);

  container.querySelectorAll("[data-taste-lift-cooldown]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        const rowIds = JSON.parse(btn.dataset.tasteLiftCooldown || "[]");
        await repairFeedback("feedback_delete", rowIds);
        setStatus("Story cooldown lifted.");
      } catch (error) {
        setStatus(error.message, true);
      }
    });
  });

  container.querySelectorAll("[data-chip-remove]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const draft = ensureLearnedDraft();
      const field = btn.dataset.chipRemove;
      const value = btn.dataset.chipValue;
      draft[field] = (draft[field] || []).filter((item) => item !== value);
      renderTasteDashboard(container);
    });
  });

  container.querySelectorAll("[data-chip-add]").forEach((btn) => {
    const field = btn.dataset.chipAdd;
    const input = container.querySelector(`[data-chip-input="${field}"]`);
    const submitChip = () => {
      const value = String(input?.value || "").trim();
      if (!value) {
        return;
      }
      const draft = ensureLearnedDraft();
      draft[field] = draft[field] || [];
      if (!draft[field].some((item) => item.toLowerCase() === value.toLowerCase())) {
        draft[field].push(value);
      }
      renderTasteDashboard(container);
    };
    btn.addEventListener("click", submitChip);
    input?.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        submitChip();
      }
    });
  });

  container.querySelectorAll("[data-weight-slider]").forEach((slider) => {
    slider.addEventListener("input", () => {
      const draft = ensureLearnedDraft();
      const field = slider.dataset.weightSlider;
      const key = slider.dataset.weightKey;
      const value = Number(slider.value);
      draft[field] = draft[field] || {};
      draft[field][key] = value;
      const badge = container.querySelector(`[data-weight-readout="${CSS.escape(field + ":" + key)}"]`);
      if (badge) {
        badge.textContent = `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
        badge.className = `taste-badge ${value >= 0 ? "positive" : "negative"}`;
      }
    });
  });

  container.querySelectorAll("[data-weight-remove]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const draft = ensureLearnedDraft();
      const field = btn.dataset.weightRemove;
      const key = btn.dataset.weightKey;
      if (draft[field]) {
        delete draft[field][key];
      }
      renderTasteDashboard(container);
    });
  });

  container.querySelectorAll("[data-weight-add]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const field = btn.dataset.weightAdd;
      const input = container.querySelector(`[data-weight-new-key="${field}"]`);
      const name = String(input?.value || "").trim();
      if (!name) {
        return;
      }
      const draft = ensureLearnedDraft();
      draft[field] = draft[field] || {};
      if (draft[field][name] === undefined) {
        draft[field][name] = 0.5;
      }
      renderTasteDashboard(container);
    });
  });

  const notesInput = byId("tasteVisualNotes");
  notesInput?.addEventListener("input", () => {
    ensureLearnedDraft().notes = notesInput.value;
  });

  const mergeSearch = byId("visualMergeSearch");
  mergeSearch?.addEventListener("input", () => {
    tasteUi.mergeQuery = mergeSearch.value;
    renderTasteDashboard(container);
    byId("visualMergeSearch")?.focus();
  });

  container.querySelectorAll("[data-merge-check]").forEach((box) => {
    box.addEventListener("change", () => {
      const key = box.dataset.mergeCheck;
      if (box.checked) {
        tasteUi.mergeSelected.add(key);
      } else {
        tasteUi.mergeSelected.delete(key);
      }
      const btn = byId("visualMergeSubmit");
      if (btn) {
        btn.textContent = `Merge Selected (${tasteUi.mergeSelected.size})`;
      }
    });
  });

  byId("visualMergeSubmit")?.addEventListener("click", submitVisualMerge);

  const splitSelect = byId("visualSplitSelect");
  splitSelect?.addEventListener("change", () => {
    tasteUi.splitSourceKey = splitSelect.value;
    tasteUi.splitCoverageIds.clear();
    tasteUi.splitFeedbackIds.clear();
    renderTasteDashboard(container);
  });

  container.querySelectorAll("[data-split-cov]").forEach((box) => {
    box.addEventListener("change", () => {
      const id = box.dataset.splitCov;
      if (box.checked) tasteUi.splitCoverageIds.add(id);
      else tasteUi.splitCoverageIds.delete(id);
    });
  });
  container.querySelectorAll("[data-split-fb]").forEach((box) => {
    box.addEventListener("change", () => {
      const id = box.dataset.splitFb;
      if (box.checked) tasteUi.splitFeedbackIds.add(id);
      else tasteUi.splitFeedbackIds.delete(id);
    });
  });
  byId("visualSplitSubmit")?.addEventListener("click", submitVisualSplit);
}

function bindFeedRowActions(scope) {
  scope.querySelectorAll("[data-taste-delete-row]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const rowId = btn.dataset.tasteDeleteRow;
      if (!rowId) return;
      try {
        await repairFeedback("feedback_delete", [rowId]);
        setStatus("Feedback removed.");
      } catch (error) {
        setStatus(error.message, true);
      }
    });
  });

  scope.querySelectorAll("[data-taste-edit-row]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const rowId = btn.dataset.tasteEditRow;
      const select = scope.querySelector(`[data-taste-action-select="${CSS.escape(rowId)}"]`);
      const nextAction = select ? select.value : "";
      if (!rowId || !nextAction) return;
      try {
        await api("/api/memory/repair", {
          method: "POST",
          body: JSON.stringify({
            action: "feedback_edit",
            row_ids: [rowId],
            event: { action: nextAction },
            confirm: true,
          }),
        });
        await refreshMemoryCallback();
        setStatus(`Updated rating to ${feedbackLabels[nextAction] || nextAction}. Click Rebuild Weights if desired.`);
      } catch (error) {
        setStatus(error.message, true);
      }
    });
  });
}

async function repairFeedback(action, rowIds) {
  await api("/api/memory/repair", {
    method: "POST",
    body: JSON.stringify({ action, row_ids: rowIds, confirm: true }),
  });
  await refreshMemoryCallback();
}

async function resetPreferences(mode) {
  try {
    setStatus(mode === "rebuild_decayed" ? "Rebuilding weights with 30-day decay..." : "Resetting learned weights...");
    const result = await api("/api/learned-preferences/reset", {
      method: "POST",
      body: JSON.stringify({ mode }),
    });
    state.learned = { path: result.path, preferences: result.preferences };
    state.learnedDraft = null;
    if (result.memory) {
      state.memory = result.memory;
    }
    await refreshMemoryCallback();
    setStatus(
      mode === "rebuild_decayed"
        ? "Rebuilt preference weights from feedback history using 30-day decay."
        : "Reset learned topic & source weights (manual lists preserved)."
    );
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function saveVisualPreferences() {
  try {
    const draft = ensureLearnedDraft();
    const saved = await api("/api/learned-preferences", {
      method: "PUT",
      body: JSON.stringify(draft),
    });
    state.learned = saved;
    state.learnedDraft = null;
    await refreshMemoryCallback();
    setStatus("Saved visual preferences.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function submitVisualMerge() {
  const keys = [...tasteUi.mergeSelected];
  if (keys.length < 2) {
    setStatus("Select at least two stories to merge.", true);
    return;
  }
  const titleInput = byId("visualMergeTitle");
  const title = String(titleInput?.value || "").trim();
  const canonical = title ? { title } : {};
  try {
    const response = await api("/api/memory/repair", {
      method: "POST",
      body: JSON.stringify({
        action: "story_merge",
        source_story_keys: keys,
        canonical_story: canonical,
        confirm: true,
      }),
    });
    tasteUi.mergeSelected.clear();
    if (titleInput) titleInput.value = "";
    state.memory = response.memory;
    await refreshMemoryCallback();
    setStatus(`Merged ${keys.length} stories cleanly.`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function submitVisualSplit() {
  const sourceKey = tasteUi.splitSourceKey;
  const covIds = [...tasteUi.splitCoverageIds];
  const fbIds = [...tasteUi.splitFeedbackIds];
  const title = String(byId("visualSplitTitle")?.value || "").trim();
  const topic = String(byId("visualSplitTopic")?.value || "").trim();
  if (!sourceKey || !covIds.length || !title) {
    setStatus("Choose a source story, check at least 1 coverage row, and enter a new story title.", true);
    return;
  }
  const newStory = { title };
  if (topic) newStory.topic = topic;
  try {
    const response = await api("/api/memory/repair", {
      method: "POST",
      body: JSON.stringify({
        action: "story_split",
        source_story_key: sourceKey,
        new_story: newStory,
        coverage_row_ids: covIds,
        feedback_row_ids: fbIds,
        confirm: true,
      }),
    });
    tasteUi.splitCoverageIds.clear();
    tasteUi.splitFeedbackIds.clear();
    state.memory = response.memory;
    await refreshMemoryCallback();
    setStatus(`Split ${covIds.length} coverage rows into "${title}".`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

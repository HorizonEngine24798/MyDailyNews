import { api } from "./api.js";
import { byId, escapeAttr, escapeHtml, setStatus } from "./dom.js";
import { state } from "./state.js";

const LAND = [
  "M55 110 L105 65 180 55 238 84 220 120 183 135 166 175 125 190 87 160 Z",
  "M205 200 L255 220 286 278 268 350 235 425 204 365 190 282 Z",
  "M447 102 L485 72 548 78 580 105 626 91 704 110 770 92 855 123 916 166 884 205 805 196 770 235 694 221 640 178 585 191 540 156 500 164 470 140 Z",
  "M477 184 L540 185 584 230 565 325 520 390 470 325 449 245 Z",
  "M792 330 L855 306 914 338 896 397 838 414 788 380 Z",
  "M927 405 L947 422 937 448 920 432 Z",
  "M351 74 L383 51 405 72 386 103 357 99 Z"
];

export async function loadMap(date = "") {
  setStatus("Loading map");
  const query = date ? `?date=${encodeURIComponent(date)}` : "";
  state.map = await api(`/api/map${query}`);
  const stories = state.map.stories || [];
  if (!stories.some((story) => story.story_id === state.currentMapStory)) {
    state.currentMapStory = (stories.find((story) => story.loci?.length) || stories[0] || {}).story_id || "";
  }
  renderMap();
  setStatus("Ready");
}

export function bindMapEvents() {
  byId("mapDate").addEventListener("change", (event) => {
    state.currentMapStory = "";
    loadMap(event.target.value).catch((error) => setStatus(error.message, true));
  });
  byId("mapCoverageToggle").addEventListener("change", (event) => {
    state.mapShowCoverage = event.target.checked;
    renderMapCanvas();
  });
  byId("mapStoryList").addEventListener("click", selectStoryFromEvent);
  byId("worldMap").addEventListener("click", selectStoryFromEvent);
  byId("worldMap").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") selectStoryFromEvent(event);
  });
}

export function renderMap() {
  const payload = state.map || { available_dates: [], stories: [], warnings: [] };
  byId("mapDate").innerHTML = (payload.available_dates || [])
    .map((date) => `<option value="${escapeAttr(date)}" ${date === payload.date ? "selected" : ""}>${escapeHtml(date)}</option>`)
    .join("");
  byId("mapDate").disabled = !payload.available_dates?.length;
  byId("mapCoverageToggle").checked = state.mapShowCoverage;
  const stories = payload.stories || [];
  byId("mapStoryCount").textContent = `${stories.length} ${stories.length === 1 ? "story" : "stories"}`;
  byId("mapWarnings").innerHTML = (payload.warnings || [])
    .map((warning) => `<div>${escapeHtml(warning)}</div>`)
    .join("");
  byId("mapStoryList").innerHTML = stories.length
    ? stories.map(renderStoryRow).join("")
    : `<div class="empty">No map stories are available.</div>`;
  renderMapCanvas();
  renderMapDetails();
}

function renderStoryRow(story) {
  const selected = story.story_id === state.currentMapStory;
  const found = story.search_summary?.found_country_count || 0;
  const searched = story.search_summary?.searched_country_count || 0;
  const location = story.loci?.length ? story.loci.map((locus) => locus.label).join(", ") : "Location unavailable";
  return `<button class="map-story-row ${selected ? "active" : ""}" type="button" data-story-id="${escapeAttr(story.story_id)}">
    <strong>${escapeHtml(story.title)}</strong>
    <span>${escapeHtml(location)}</span>
    <small>${found} of ${searched} searched countries returned coverage</small>
  </button>`;
}

function renderMapCanvas() {
  const svg = byId("worldMap");
  const stories = state.map?.stories || [];
  const selected = stories.find((story) => story.story_id === state.currentMapStory);
  const land = LAND.map((path) => `<path class="map-land" d="${path}"></path>`).join("");
  const graticule = [100, 200, 300, 400]
    .map((y) => `<line class="map-gridline" x1="0" y1="${y}" x2="1000" y2="${y}"></line>`)
    .concat([167, 333, 500, 667, 833].map((x) => `<line class="map-gridline" x1="${x}" y1="0" x2="${x}" y2="500"></line>`))
    .join("");
  const coverage = state.mapShowCoverage && selected
    ? (selected.coverage_points || []).map(renderCoveragePoint).join("")
    : "";
  const pins = stories.flatMap((story) => (story.loci || []).map((locus) => renderStoryPoint(story, locus))).join("");
  svg.innerHTML = `<rect class="map-ocean" width="1000" height="500"></rect>${graticule}${land}${coverage}${pins}`;
}

function renderStoryPoint(story, locus) {
  const [x, y] = project(locus.lat, locus.lon);
  const selected = story.story_id === state.currentMapStory;
  return `<g class="map-marker story-marker ${selected ? "selected" : ""}" data-story-id="${escapeAttr(story.story_id)}" tabindex="0">
    <title>${escapeHtml(`${story.title} — ${locus.label}`)}</title>
    <circle cx="${x}" cy="${y}" r="${selected ? 10 : 7}"></circle>
    <path d="M ${x} ${y + 5} L ${x - 5} ${y + 17} L ${x + 5} ${y + 17} Z"></path>
  </g>`;
}

function renderCoveragePoint(point) {
  const [x, y] = project(point.lat, point.lon);
  const label = point.status === "found"
    ? `${point.label}: ${point.article_count} retained article(s)`
    : `${point.label}: searched, no articles retained`;
  return `<g class="map-marker coverage-marker ${point.status}" tabindex="0">
    <title>${escapeHtml(label)}</title>
    <circle cx="${x}" cy="${y}" r="${point.status === "found" ? 8 : 6}"></circle>
  </g>`;
}

function renderMapDetails() {
  const story = (state.map?.stories || []).find((item) => item.story_id === state.currentMapStory);
  const target = byId("mapDetails");
  if (!story) {
    target.innerHTML = `<div class="empty">Select a story to inspect its reporting footprint.</div>`;
    return;
  }
  const summary = story.search_summary || {};
  const quality = story.coverage_quality || {};
  const points = story.coverage_points || [];
  const found = points.filter((point) => point.status === "found");
  const empty = points.filter((point) => point.status === "searched_empty");
  target.innerHTML = `<div class="pane-header"><h3>${escapeHtml(story.title)}</h3></div>
    <div class="map-details-body">
      ${story.summary ? `<p>${escapeHtml(story.summary)}</p>` : ""}
      <div class="map-stats">
        <div><strong>${summary.selected_source_count || 0}</strong><span>sources selected</span></div>
        <div><strong>${summary.found_country_count || 0}</strong><span>countries found</span></div>
        <div><strong>${summary.searched_empty_country_count || 0}</strong><span>searched empty</span></div>
      </div>
      ${renderCountryGroup("Coverage found", found)}
      ${renderCountryGroup("Searched, nothing retained", empty)}
      <p class="muted small">Countries not listed were not searched for this story. Search scope comes from the perspectives planner, so absence is not evidence of no reporting.</p>
      ${renderProviderStatuses(summary.provider_statuses)}
      ${quality.status ? `<p><strong>Coverage quality:</strong> ${escapeHtml(quality.status)}</p>` : ""}
      ${story.framing_summary ? `<h4>Framing synthesis</h4><p>${escapeHtml(story.framing_summary)}</p>` : ""}
      ${renderArticles(story.coverage_articles || [])}
      ${(story.warnings || []).map((warning) => `<div class="map-inline-warning">${escapeHtml(warning)}</div>`).join("")}
    </div>`;
}

function renderCountryGroup(title, points) {
  if (!points.length) return "";
  return `<h4>${escapeHtml(title)}</h4><div class="map-country-list">${points.map((point) =>
    `<div><strong>${escapeHtml(point.label)}</strong><span>${point.article_count ? `${point.article_count} article(s)` : (point.sources || []).join(", ") || "No retained result"}</span></div>`
  ).join("")}</div>`;
}

function renderArticles(articles) {
  if (!articles.length) return "";
  return `<details class="map-articles"><summary>${articles.length} coverage articles</summary>${articles.map((article) => {
    const title = escapeHtml(article.title || "Untitled article");
    const url = safeUrl(article.url);
    const headline = url
      ? `<a href="${escapeAttr(url)}" target="_blank" rel="noreferrer">${title}</a>`
      : title;
    const source = [article.source, article.country].filter(Boolean).join(" · ");
    return `<div>${headline}<span>${escapeHtml(source)}</span></div>`;
  }).join("")}</details>`;
}

function renderProviderStatuses(statuses) {
  if (!statuses || typeof statuses !== "object") return "";
  const rows = Object.entries(statuses).map(([name, value]) => {
    const status = value && typeof value === "object" ? value.status : value;
    return `${name}: ${status || "unknown"}`;
  });
  return rows.length ? `<p class="muted small"><strong>Retrieval providers:</strong> ${escapeHtml(rows.join(" · "))}</p>` : "";
}

function safeUrl(value) {
  const url = String(value || "").trim();
  return /^https?:\/\//i.test(url) ? url : "";
}

function selectStoryFromEvent(event) {
  const marker = event.target.closest("[data-story-id]");
  if (!marker) return;
  state.currentMapStory = marker.dataset.storyId;
  renderMap();
}

function project(lat, lon) {
  return [((Number(lon) + 180) / 360) * 1000, ((90 - Number(lat)) / 180) * 500];
}

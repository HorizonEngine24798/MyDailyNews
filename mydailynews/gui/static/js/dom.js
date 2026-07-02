import { monthNames } from "./state.js";

export function byId(id) {
  return document.getElementById(id);
}

export function setStatus(text, isError = false) {
  const target = byId("statusLine");
  target.textContent = text || "";
  target.style.color = isError ? "var(--danger)" : "var(--muted)";
}

export function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function escapeAttr(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}

export function linkify(value) {
  return String(value).replace(/https?:\/\/[^\s<]+/g, (url) => {
    const clean = url.replace(/[),.]+$/, "");
    const suffix = url.slice(clean.length);
    return `<a href="${clean}" target="_blank" rel="noreferrer">${clean}</a>${suffix}`;
  });
}

export function renderMarkdown(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const html = [];
  let inList = false;

  function closeList() {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  }

  lines.forEach((line) => {
    const text = line.trim();
    if (!text) {
      closeList();
      return;
    }
    if (text.startsWith("### ")) {
      closeList();
      html.push(`<h3>${linkify(escapeHtml(text.slice(4)))}</h3>`);
    } else if (text.startsWith("## ")) {
      closeList();
      html.push(`<h2>${linkify(escapeHtml(text.slice(3)))}</h2>`);
    } else if (text.startsWith("# ")) {
      closeList();
      html.push(`<h1>${linkify(escapeHtml(text.slice(2)))}</h1>`);
    } else if (text.startsWith("- ")) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${linkify(escapeHtml(text.slice(2)))}</li>`);
    } else {
      closeList();
      html.push(`<p>${linkify(escapeHtml(text))}</p>`);
    }
  });
  closeList();
  return html.join("");
}

export function humanLabel(label) {
  return String(label || "")
    .replace(/^\[(\d+)\]$/, "Item $1")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

export function shouldUseTextarea(label, text) {
  const key = String(label || "").toLowerCase();
  return (
    text.length > 72 ||
    [
      "description",
      "briefing_style",
      "custom_instructions",
      "editorial_style",
      "portfolio_or_stake_notes",
      "notes",
      "url",
    ].includes(key)
  );
}

export function ordinal(day) {
  if (!day) {
    return "Undated";
  }
  const mod100 = day % 100;
  if (mod100 >= 11 && mod100 <= 13) {
    return `${day}th`;
  }
  const suffix = { 1: "st", 2: "nd", 3: "rd" }[day % 10] || "th";
  return `${day}${suffix}`;
}

export function reportDisplayTitle(report) {
  const parts = String(report.date || "").split("-");
  const month = monthNames[Number(parts[1] || 0) - 1] || "";
  const day = Number(parts[2] || 0);
  const dateLabel = month && day ? `${month} ${ordinal(day)}` : String(report.date || "").trim();
  const typeLabel = humanReportType(report.kind || report.brief_name || "report");
  return [dateLabel, typeLabel].filter(Boolean).join(" ") || typeLabel;
}

export function reportListTitle(report) {
  const typeLabel = humanReportType(report.kind || report.brief_name || "report");
  const dateLabel = String(report.date || "").trim();
  return [typeLabel, dateLabel].filter(Boolean).join(" ") || typeLabel;
}

export function humanReportType(kind) {
  const normalized = String(kind || "report")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
  return normalized || "Report";
}

import { byId, escapeHtml, humanLabel, shouldUseTextarea } from "./dom.js";

export function renderForm(hostId, getRoot) {
  const host = byId(hostId);
  host.innerHTML = "";

  const grid = document.createElement("div");
  grid.className = "form-grid";
  renderValue(grid, getRoot(), [], "root", getRoot, () => renderForm(hostId, getRoot));
  host.appendChild(grid);
}

function renderValue(parent, value, path, label, getRoot, rerender) {
  if (Array.isArray(value)) {
    renderArray(parent, value, path, label, getRoot, rerender);
    return;
  }
  if (value && typeof value === "object") {
    renderObject(parent, value, path, label, getRoot, rerender);
    return;
  }
  renderScalar(parent, value, path, label, getRoot);
}

function renderObject(parent, value, path, label, getRoot, rerender) {
  const isRoot = path.length === 0;
  const card = document.createElement("div");
  card.className = isRoot ? "form-node" : "form-card";

  if (!isRoot) {
    const header = document.createElement("div");
    header.className = "form-card-header";
    header.innerHTML = `<h4>${escapeHtml(humanLabel(label))}</h4>`;
    if (canAddField(path)) {
      const add = document.createElement("button");
      add.type = "button";
      add.className = "mini-button";
      add.textContent = "Add field";
      add.addEventListener("click", () => addObjectField(getRoot(), path, rerender));
      header.appendChild(add);
    }
    card.appendChild(header);
  }

  Object.keys(value).forEach((key) => {
    renderValue(card, value[key], path.concat(key), key, getRoot, rerender);
  });

  if (Object.keys(value).length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No fields.";
    card.appendChild(empty);
  }
  parent.appendChild(card);
}

function renderArray(parent, value, path, label, getRoot, rerender) {
  const card = document.createElement("div");
  card.className = "form-card";

  const header = document.createElement("div");
  header.className = "form-card-header";
  header.innerHTML = `<h4>${escapeHtml(humanLabel(label))}</h4><span class="muted small">${value.length} item(s)</span>`;
  card.appendChild(header);

  value.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "form-card";

    const rowHeader = document.createElement("div");
    rowHeader.className = "form-card-header";
    rowHeader.innerHTML = `<span class="field-label">Item ${index + 1}</span>`;

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "mini-button";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      const array = getPath(getRoot(), path);
      array.splice(index, 1);
      rerender();
    });

    rowHeader.appendChild(remove);
    row.appendChild(rowHeader);
    renderValue(row, item, path.concat(index), `[${index + 1}]`, getRoot, rerender);
    card.appendChild(row);
  });

  const actions = document.createElement("div");
  actions.className = "array-actions";

  const add = document.createElement("button");
  add.type = "button";
  add.className = "mini-button";
  add.textContent = "Add item";
  add.addEventListener("click", () => {
    const array = getPath(getRoot(), path);
    array.push(emptyArrayItem(array));
    rerender();
  });

  actions.appendChild(add);
  card.appendChild(actions);
  parent.appendChild(card);
}

function renderScalar(parent, value, path, label, getRoot) {
  const row = document.createElement("label");
  row.className = "field-row";
  row.innerHTML = `<span class="field-label">${escapeHtml(humanLabel(label))}</span>`;

  if (typeof value === "boolean") {
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = value;
    input.addEventListener("change", () => setPath(getRoot(), path, input.checked));
    row.appendChild(input);
  } else if (typeof value === "number") {
    const input = document.createElement("input");
    input.type = "number";
    input.step = Number.isInteger(value) ? "1" : "any";
    input.className = "field-input";
    input.value = String(value);
    input.addEventListener("input", () => {
      const parsed = input.step === "1" ? parseInt(input.value || "0", 10) : parseFloat(input.value || "0");
      setPath(getRoot(), path, Number.isFinite(parsed) ? parsed : 0);
    });
    row.appendChild(input);
  } else {
    const text = value == null ? "" : String(value);
    const input = document.createElement(shouldUseTextarea(label, text) ? "textarea" : "input");
    input.className = input.tagName === "TEXTAREA" ? "field-textarea" : "field-input";
    input.value = text;
    input.addEventListener("input", () => setPath(getRoot(), path, input.value));
    row.appendChild(input);
  }

  parent.appendChild(row);
}

function canAddField(path) {
  const last = String(path[path.length - 1] || "");
  return ["beats", "topic_weights", "source_weights"].includes(last);
}

function addObjectField(root, path, rerender) {
  const key = window.prompt("Field name");
  if (!key) {
    return;
  }

  const object = getPath(root, path);
  if (!object || typeof object !== "object" || Array.isArray(object)) {
    return;
  }
  if (Object.prototype.hasOwnProperty.call(object, key)) {
    return;
  }

  const last = String(path[path.length - 1] || "");
  object[key] = last.includes("weight") || last === "beats" ? 0 : "";
  rerender();
}

function emptyArrayItem(array) {
  return array.length ? emptyFromSample(array[0]) : "";
}

function emptyFromSample(sample) {
  if (Array.isArray(sample)) {
    return [];
  }
  if (sample && typeof sample === "object") {
    const output = {};
    Object.keys(sample).forEach((key) => {
      output[key] = emptyFromSample(sample[key]);
    });
    return output;
  }
  if (typeof sample === "boolean") {
    return false;
  }
  if (typeof sample === "number") {
    return 0;
  }
  return "";
}

function getPath(root, path) {
  return path.reduce((current, key) => current[key], root);
}

function setPath(root, path, value) {
  if (!path.length) {
    return;
  }
  const last = path[path.length - 1];
  const parent = getPath(root, path.slice(0, -1));
  parent[last] = value;
}

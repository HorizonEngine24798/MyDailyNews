import { byId, escapeHtml, humanLabel, shouldUseTextarea } from "./dom.js";

const detailsOpen = {};

export function renderForm(hostId, getRoot) {
  const host = byId(hostId);
  host.innerHTML = "";

  const grid = document.createElement("div");
  grid.className = "form-grid";
  renderValue(grid, getRoot(), [], "", getRoot, () => renderForm(hostId, getRoot), hostId);
  host.appendChild(grid);
}

function renderValue(parent, value, path, label, getRoot, rerender, hostId) {
  if (Array.isArray(value)) {
    renderArray(parent, value, path, label, getRoot, rerender, hostId);
    return;
  }
  if (value && typeof value === "object") {
    renderObject(parent, value, path, label, getRoot, rerender, hostId);
    return;
  }
  renderScalar(parent, value, path, label, getRoot);
}

function renderObject(parent, value, path, label, getRoot, rerender, hostId) {
  const isRoot = path.length === 0;
  const card = document.createElement(isRoot ? "div" : "details");
  card.className = isRoot ? "form-node" : "form-card";

  if (!isRoot) {
    restoreDetailsOpen(card, hostId, path);
    const header = document.createElement("summary");
    header.className = "form-card-header";
    header.innerHTML = `<h4>${escapeHtml(humanLabel(label))}</h4>`;
    card.appendChild(header);

    if (canAddField(path)) {
      const actions = document.createElement("div");
      actions.className = "array-actions";
      const add = document.createElement("button");
      add.type = "button";
      add.className = "mini-button";
      add.textContent = "Add field";
      add.addEventListener("click", () => addObjectField(getRoot(), path, rerender));
      actions.appendChild(add);
      card.appendChild(actions);
    }
  }

  Object.keys(value).forEach((key) => {
    renderValue(card, value[key], path.concat(key), key, getRoot, rerender, hostId);
  });

  if (Object.keys(value).length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No fields.";
    card.appendChild(empty);
  }
  parent.appendChild(card);
}

function renderArray(parent, value, path, label, getRoot, rerender, hostId) {
  const isRoot = path.length === 0;
  const card = document.createElement(isRoot ? "div" : "details");
  card.className = isRoot ? "form-node" : "form-card";

  if (!isRoot) {
    restoreDetailsOpen(card, hostId, path);
    const header = document.createElement("summary");
    header.className = "form-card-header";
    header.innerHTML = `<h4>${escapeHtml(humanLabel(label))}</h4><span class="muted small">${value.length} item(s)</span>`;
    card.appendChild(header);
  }

  value.forEach((item, index) => {
    const itemPath = path.concat(index);
    const remove = removeArrayButton(getRoot, path, index, rerender);

    if (!item || typeof item !== "object") {
      const row = document.createElement("div");
      row.className = "array-item-row";
      renderScalar(row, item, itemPath, "", getRoot);
      row.appendChild(remove);
      card.appendChild(row);
      return;
    }

    const row = document.createElement("details");
    row.className = "form-card";
    restoreDetailsOpen(row, hostId, itemPath);

    const rowHeader = document.createElement("summary");
    rowHeader.className = "form-card-header";
    rowHeader.innerHTML = `<h4>${escapeHtml(arrayItemLabel(item, index))}</h4>`;
    row.appendChild(rowHeader);

    const rowActions = document.createElement("div");
    rowActions.className = "array-actions";
    rowActions.appendChild(remove);
    row.appendChild(rowActions);
    renderArrayItem(row, item, itemPath, arrayItemLabel(item, index), getRoot, rerender, hostId);
    card.appendChild(row);
  });

  if (!value.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No items.";
    card.appendChild(empty);
  }

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

function renderArrayItem(parent, value, path, label, getRoot, rerender, hostId) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    Object.keys(value).forEach((key) => {
      renderValue(parent, value[key], path.concat(key), key, getRoot, rerender, hostId);
    });
    if (Object.keys(value).length === 0) {
      const empty = document.createElement("div");
      empty.className = "empty";
      empty.textContent = "No fields.";
      parent.appendChild(empty);
    }
    return;
  }
  renderValue(parent, value, path, label, getRoot, rerender, hostId);
}

function renderScalar(parent, value, path, label, getRoot) {
  const row = document.createElement("label");
  row.className = "field-row";
  row.innerHTML = label ? `<span class="field-label">${escapeHtml(humanLabel(label))}</span>` : "";

  if (typeof value === "boolean") {
    row.classList.add("boolean-row");
    const input = document.createElement("select");
    input.className = "field-input boolean-input";
    input.innerHTML = `<option value="true">true</option><option value="false">false</option>`;
    input.value = value ? "true" : "false";
    input.addEventListener("change", () => setPath(getRoot(), path, input.value === "true"));
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

function removeArrayButton(getRoot, path, index, rerender) {
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "mini-button";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => {
    const array = getPath(getRoot(), path);
    array.splice(index, 1);
    rerender();
  });
  return remove;
}

function arrayItemLabel(value, index) {
  return arrayItemText(value) || `Entry ${index + 1}`;
}

function arrayItemText(value) {
  if (value == null) {
    return "";
  }
  if (Array.isArray(value)) {
    return value.map(arrayItemText).filter(Boolean).join(", ");
  }
  if (typeof value !== "object") {
    return String(value).trim();
  }

  const preferred = ["title", "headline", "name", "label", "source", "topic", "story_key", "id", "url"];
  for (const key of preferred) {
    const text = arrayItemText(value[key]);
    if (text) {
      return text;
    }
  }
  for (const nested of Object.values(value)) {
    const text = arrayItemText(nested);
    if (text) {
      return text;
    }
  }
  return "";
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

function restoreDetailsOpen(node, hostId, path) {
  const key = detailsKey(hostId, path);
  node.open = Object.prototype.hasOwnProperty.call(detailsOpen, key) ? detailsOpen[key] : true;
  node.addEventListener("toggle", (event) => {
    if (event.target === node) {
      detailsOpen[key] = node.open;
    }
  });
}

function detailsKey(hostId, path) {
  return `${hostId}:${JSON.stringify(path)}`;
}

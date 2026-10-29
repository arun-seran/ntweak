"use strict";

const $ = (id) => document.getElementById(id);
const KEY = "ntweak.workspace.v1";
const DEFAULT_RECIPE = {
  name: "My first experiment", description: "", model: "Qwen/Qwen3.5-4B", source: "", neurons: [],
  sampling: { temperature: 0.7, top_p: 0.95, min_p: 0.05, repetition_penalty: 1.1, max_new_tokens: 512, seed: 42 }
};
const samplingFields = {temperature: "temperature", top_p: "topP", min_p: "minP", repetition_penalty: "repetitionPenalty", max_new_tokens: "maxTokens", seed: "seed"};
let state = { recipe: structuredClone(DEFAULT_RECIPE), saved: [], turns: [], steering: true, compare: false, thinking: false, maxContext: 4096, systemPrompt: "" };
try {
  const stored = JSON.parse(localStorage.getItem(KEY));
  if (stored && Array.isArray(stored.turns) && Array.isArray(stored.saved) && Array.isArray(stored.recipe?.neurons)) {
    state = { ...state, ...stored, recipe: { ...DEFAULT_RECIPE, ...stored.recipe, sampling: {...DEFAULT_RECIPE.sampling, ...stored.recipe.sampling} } };
  }
} catch { /* A fresh workspace is safe when storage is unavailable or corrupt. */ }
let status = null, inspected = null, busy = false, preparing = false, activeRequest = null, stopping = false;
let importMode = "link", toastTimer, persistTimer, storageWarned = false;
const seenJobs = new Set();
const terminal = (job) => ["done", "error", "cancelled"].includes(job.status);

function node(tag, props = {}, children = []) {
  const out = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "text") out.textContent = String(value);
    else if (key === "class") out.className = value;
    else if (key.startsWith("on")) out.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key in out && key !== "list") out[key] = value;
    else out.setAttribute(key, value);
  }
  for (const child of children) out.append(child);
  return out;
}
function toast(text) {
  $("toast").textContent = text; $("toast").hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 4200);
}
function persist() {
  clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    try { localStorage.setItem(KEY, JSON.stringify(cleanWorkspace())); }
    catch { if (!storageWarned) { storageWarned = true; toast("Browser storage is full or unavailable. Export your chat and recipes to keep them."); } }
  }, 150);
}
function showError(id, error) { $(id).textContent = error?.message || String(error || ""); $(id).hidden = !error; }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function bytes(value) {
  if (!Number.isFinite(value) || value === 0) return "0 B";
  const n = Math.min(4, Math.floor(Math.log(value) / Math.log(1024)));
  return `${(value / 1024 ** n).toFixed(n > 1 ? 2 : 0)} ${["B", "KB", "MB", "GB", "TB"][n]}`;
}
function downloadJSON(value, name) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob), link = node("a", { href: url, download: name });
  document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function slug(text) { return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "recipe"; }
function openDialog(id) { if (!$(id).open) $(id).showModal(); }
function activeRows() { return state.steering ? state.recipe.neurons.filter(x => x.enabled !== false) : []; }

function syncRecipeFields() {
  $("recipeName").value = state.recipe.name;
  $("recipeModel").value = state.recipe.model;
  $("steeringEnabled").checked = state.steering;
  for (const [key, id] of Object.entries(samplingFields)) $(id).value = state.recipe.sampling[key];
  $("compare").checked = state.compare;
  $("thinking").checked = state.thinking;
  $("maxContext").value = state.maxContext;
  $("systemPrompt").value = state.systemPrompt;
  renderTweaks(); renderSaved();
}
function renderSaved() {
  $("savedRecipes").replaceChildren(node("option", {value: "", text: "Saved recipes…"}), ...state.saved.map((r, i) => node("option", { value: String(i), text: r.name })));
}
function updateSummary() {
  const count = activeRows().length;
  $("tweakCount").textContent = state.recipe.neurons.length;
  $("steeringSummary").replaceChildren(node("i", {class: "small-dot"}), document.createTextNode(state.steering ? (count ? `${count} active tweak${count === 1 ? "" : "s"} · every token` : "No active tweaks") : "Steering bypassed · baseline"));
}
function renderTweaks() {
  const list = $("tweakList"); list.replaceChildren();
  if (!state.recipe.neurons.length) list.append(node("div", {class: "empty-tweaks"}, [node("span", {text: "≋", "aria-hidden": "true"}), document.createTextNode("Your next experiment starts here."), node("br"), document.createTextNode("Add a neuron or bring a recipe.")]));
  state.recipe.neurons.forEach((row, i) => {
    const card = node("div", { class: `tweak-card${row.enabled === false ? " disabled" : ""}` });
    const enabled = node("input", { type: "checkbox", checked: row.enabled !== false, "aria-label": `Enable tweak ${i + 1}`, onChange: () => { row.enabled = enabled.checked; card.classList.toggle("disabled", !row.enabled); updateSummary(); persist(); } });
    const caption = node("span", {text: row.label || `NEURON ${String(i + 1).padStart(2, "0")}`});
    const remove = node("button", {class: "icon-button remove-tweak", text: "×", "aria-label": `Remove tweak ${i + 1}`, onClick: () => { state.recipe.neurons.splice(i, 1); renderTweaks(); persist(); }});
    card.append(node("div", {class: "tweak-card-header"}, [node("label", {class: "neuron-label"}, [enabled, caption]), remove]));
    const fields = node("div", {class: "tweak-fields"});
    let doseInput, slider;
    for (const [key, title, min, max, step] of [["layer", "LAYER", 0, 1000, 1], ["neuron", "NEURON", 0, 1000000, 1], ["dose", "DOSE", -100, 100, 0.05]]) {
      const input = node("input", {type: "number", min, max, step, value: row[key], "aria-label": `${title.toLowerCase()} for tweak ${i + 1}`, onInput: () => { row[key] = input.value === "" ? null : Number(input.value); if (key === "dose" && slider) slider.value = Math.max(-3, Math.min(3, row.dose)); persist(); }});
      fields.append(node("label", {text: title}, [input]));
      if (key === "dose") doseInput = input;
    }
    card.append(fields);
    slider = node("input", {type: "range", min: -3, max: 3, step: 0.05, value: Math.max(-3, Math.min(3, row.dose)), "aria-label": `Adjust dose for tweak ${i + 1}`, onInput: () => { row.dose = Number(slider.value); doseInput.value = row.dose; persist(); }});
    card.append(node("div", {class: "dose-control"}, [node("span", {text: "−3"}), slider, node("span", {text: "+3"})]));
    const op = node("select", {"aria-label": `Operation for tweak ${i + 1}`, onChange: () => { row.mode = op.value; persist(); }}, [node("option", {value: "add", text: "Add · normalized"}), node("option", {value: "mul", text: "Multiply activation"})]);
    op.value = row.mode || "add";
    card.append(node("div", {class: "mode-line"}, [node("span", {text: "BEFORE DOWN_PROJ"}), op])); list.append(card);
  });
  updateSummary();
}
function addTweak() {
  if (state.recipe.neurons.length >= 128) return toast("A recipe can contain up to 128 neuron tweaks.");
  state.recipe.neurons.push({layer: 22, neuron: 0, dose: 0.8, mode: "add", enabled: true, label: ""});
  renderTweaks(); persist();
}
function setRecipe(recipe) {
  state.recipe = structuredClone(recipe); syncRecipeFields(); persist();
  if (status?.runtime.model && recipe.model !== status.runtime.model.repo) toast(`This recipe targets ${recipe.model}; load the same model before steering.`);
}

function renderModels() {
  if (!status) return;
  const rt = status.runtime;
  const ready = rt.state === "ready" || rt.state === "generating";
  $("connection").classList.remove("offline");
  $("connection").replaceChildren(node("i"), document.createTextNode(ready ? `${(rt.device || "local").toUpperCase()} · LOCAL` : "LOCAL SERVER"));
  $("activeModel").textContent = rt.model?.repo || "Qwen3.5 · 4B";
  $("modelSubline").textContent = ready ? `${rt.device.toUpperCase()} · ${rt.layers ? `${rt.layers} layers · ` : ""}${rt.steering_supported ? "neuron steering ready" : "chat only"}` : (rt.state === "loading" ? "Loading local weights…" : "Choose a local model to begin");
  $("send").disabled = busy || preparing || !ready;
  $("send").title = ready ? "Send message" : "Load a model in the Models panel first";
  const deps = status.dependencies;
  $("dependencyNotice").hidden = deps.ready;
  if (!deps.ready) $("dependencyNotice").replaceChildren(document.createTextNode("The UI and downloader are ready. Install the inference runtime, then restart the app with uv run python -m ntweak. In the app folder, run:"), node("br"), node("code", {text: "uv sync"}));
  const local = $("localModels"); local.replaceChildren();
  if (!status.models.length) local.append(node("div", {class: "library-empty", text: "No local models yet. Download one below to get started."}));
  const loading = status.jobs.some(j => j.kind === "load" && !terminal(j));
  for (const model of status.models) {
    const loaded = rt.model?.id === model.id;
    const loadButton = node("button", {class: "button secondary", text: "Load model", disabled: !model.available || loading || busy || !deps.ready, onClick: async () => {
      showError("modelError", null); loadButton.disabled = true;
      try { await api("/api/models/load", {id: model.id, device: $("device").value}); await refresh(); }
      catch (error) { showError("modelError", error); loadButton.disabled = false; }
    }});
    local.append(node("div", {class: "local-model"}, [node("div", {}, [node("strong", {text: model.repo}), node("small", {text: `${model.source} · ${model.available ? model.revision.slice(0, 12) : "folder unavailable"}`}), node("small", {text: model.path})]), loaded ? node("span", {class: "loaded"}, [node("i", {class: "small-dot"}), document.createTextNode("Loaded")]) : loadButton]));
  }
  $("unloadModel").hidden = !rt.model; $("unloadModel").disabled = busy;
  $("dataPath").textContent = `Models and download state: ${status.data_dir}\nConversations and saved recipes: this browser's local storage.`;
  renderJobs();
  if (inspected) updateSize();
}
function renderJobs() {
  const container = $("downloadJobs"); container.replaceChildren();
  for (const job of status.jobs.slice(-5).reverse()) {
    if (job.status === "done" && seenJobs.has(job.id)) continue;
    const done = terminal(job);
    const title = job.kind === "load" ? (job.status === "done" ? "Model loaded" : "Loading model") : (job.status === "done" ? "Download complete" : job.status === "cancelled" ? "Download paused" : "Model download");
    const controls = node("div", {class: "job-header"}, [node("strong", {text: job.status === "error" ? "Operation failed" : title})]);
    if (!done && job.kind === "download") controls.append(node("button", {class: "text-button", text: "Cancel", onClick: async () => { try { await api("/api/jobs/cancel", {id: job.id}); toast("Stopping download. Partial files will be kept."); } catch (e) { showError("modelError", e); } }}));
    const card = node("div", {class: `job ${job.status === "error" ? "error" : ""}`}, [controls]);
    if (job.total_bytes) {
      card.append(node("progress", {max: job.total_bytes, value: job.completed_bytes || 0}));
      card.append(node("small", {text: `${bytes(job.completed_bytes || 0)} / ${bytes(job.total_bytes)} · ${job.files_done || 0} / ${job.files_total} files`}));
    }
    card.append(node("small", {text: job.error || job.message || job.file || (done ? "Ready in your local library." : "Preparing…")}));
    if (done) card.append(node("button", {class: "text-button", text: "Dismiss", onClick: () => { seenJobs.add(job.id); renderJobs(); }}));
    container.append(card);
  }
}
async function refresh() {
  try {
    const next = await api("/api/status");
    const previous = status;
    status = next;
    if (previous) for (const job of next.jobs) {
      const old = previous.jobs.find(x => x.id === job.id);
      if (job.status === "done" && old?.status !== "done") toast(job.kind === "load" ? "Model loaded. Your local playground is ready." : "Download finished. Load the model to start chatting.");
    }
    renderModels();
  } catch {
    $("connection").classList.add("offline");
    $("connection").replaceChildren(node("i"), document.createTextNode("SERVER OFFLINE"));
    $("send").disabled = true;
  }
}
async function inspectRepo() {
  showError("modelError", null); $("inspectButton").disabled = true; $("inspectQwen").disabled = true;
  $("inspectButton").textContent = "Inspecting…"; inspected = null; $("downloadFiles").hidden = true;
  try {
    inspected = await api("/api/models/inspect", {repo: $("repoId").value.trim(), revision: $("revision").value.trim(), token: $("hfToken").value.trim()});
    $("fileSummary").textContent = `${inspected.files.length} files · ${inspected.commit.slice(0, 8)}`;
    renderFileList(); $("downloadFiles").hidden = false;
  } catch (error) { showError("modelError", error); }
  finally { $("inspectButton").disabled = false; $("inspectButton").textContent = "Inspect files"; $("inspectQwen").disabled = false; }
}
function renderFileList() {
  $("fileList").replaceChildren(...inspected.files.map(file => {
    const checkbox = node("input", {type: "checkbox", checked: file.selected, onChange: () => { file.selected = checkbox.checked; updateSize(); }});
    return node("label", {class: "file-row"}, [checkbox, node("span", {text: file.name}), node("small", {text: bytes(file.size)})]);
  })); updateSize();
}
function updateSize() {
  const selected = inspected.files.filter(x => x.selected);
  $("downloadSize").textContent = `${bytes(selected.reduce((sum, x) => sum + x.size, 0))} · ${selected.length} files selected`;
  $("downloadButton").disabled = !selected.length || !!status?.jobs.some(x => x.kind === "download" && !terminal(x));
}
async function startDownload() {
  if (!inspected) return;
  $("downloadButton").disabled = true; showError("modelError", null);
  try {
    await api("/api/models/download", {repo: inspected.repo, revision: inspected.commit, files: inspected.files.filter(x => x.selected).map(x => x.name), token: $("hfToken").value.trim()});
    $("hfToken").value = ""; await refresh();
  } catch (error) { showError("modelError", error); }
  finally { updateSize(); }
}

function scrollChat() { const pane = $("chatScroll"); pane.scrollTop = pane.scrollHeight; }
function createResponse(branch, title, baseline) {
  const content = node("div", {class: `message-text${branch.text ? "" : " pending"}`, text: branch.text || branch.error || "Waiting for the local model…"});
  const stats = node("div", {class: "response-stats"});
  const details = node("details", {hidden: !branch.calibration?.neurons?.length}, [node("summary", {text: "View applied doses ↗"})]);
  const card = node("div", {class: `response${baseline ? " baseline" : ""}`}, [node("div", {class: "response-header"}, [node("span", {class: "response-icon", text: baseline ? "○" : "n"}), node("strong", {text: title}), node("span", {class: "response-badge", text: baseline ? "UNSTEERED" : (branch.tweaks ? `${branch.tweaks} TWEAK${branch.tweaks === 1 ? "" : "S"}` : "UNSTEERED")})]), content, stats, details]);
  branch.view = {card, content, stats, details};
  updateBranch(branch);
  return card;
}
function updateBranch(branch) {
  const view = branch.view;
  if (!view) return;
  view.content.textContent = branch.text || branch.error || (branch.done ? (branch.cancelled ? "Stopped before any text was generated." : "No text returned.") : "Waiting for the local model…");
  view.content.classList.toggle("pending", !branch.text);
  view.content.classList.toggle("inline-error", !!branch.error);
  const meta = branch.meta;
  view.stats.textContent = meta ? `${meta.output_tokens ?? 0} tokens · ${meta.tokens_per_second ?? 0} tok/s${meta.cancelled ? " · stopped" : ""} · ${meta.model || "local"}` : (branch.error ? "Response failed. You can retry your prompt." : branch.cancelled ? "Stopped" : "");
  view.details.hidden = !branch.calibration?.neurons?.length;
  if (branch.calibration?.neurons?.length) {
    view.details.replaceChildren(node("summary", {text: "View applied doses ↗"}), ...branch.calibration.neurons.map(x => node("div", {class: "calibration-row", text: x.mode === "mul" ? `L${x.layer} N${x.neuron} × ${x.dose}` : `L${x.layer} N${x.neuron}\ndose ${x.dose} · median ${x.residual_median?.toFixed(4) ?? "—"}\n‖W[:,n]‖ ${x.column_norm?.toFixed(4)} · Δactivation ${x.activation_delta?.toFixed(4)}`})));
  }
}
function appendTurn(turn) {
  const wrapper = node("article", {class: "turn"});
  wrapper.append(node("div", {class: "user-message"}, [node("span", {class: "user-caption", text: "YOU"}), node("div", {class: "message-text", text: turn.prompt})]));
  const responses = node("div", {class: `responses${turn.baseline ? " comparing" : ""}`});
  if (turn.baseline) responses.append(createResponse(turn.baseline, "Baseline", true));
  responses.append(createResponse(turn.steered, turn.steering ? "Steered" : "Local model", false));
  wrapper.append(responses); $("messages").append(wrapper);
}
function renderConversation() {
  $("messages").replaceChildren(); $("welcome").hidden = state.turns.length > 0;
  for (const turn of state.turns) appendTurn(turn);
  scrollChat();
}
function historyFor(branch, turns) {
  const messages = state.systemPrompt.trim() ? [{role: "system", content: state.systemPrompt.trim()}] : [];
  for (const turn of turns) {
    const reply = turn[branch] || turn.steered;
    if (reply.text && !reply.error) messages.push({role: "user", content: turn.prompt}, {role: "assistant", content: reply.text});
  }
  return messages;
}
function cleanWorkspace() {
  // DOM references belong to the view, never browser storage or exported JSON.
  return JSON.parse(JSON.stringify(state, (key, value) => key === "view" ? undefined : value));
}
// Serialize only data: DOM nodes would otherwise become empty objects in stored conversations.
function saveImmediately() {
  clearTimeout(persistTimer);
  try { localStorage.setItem(KEY, JSON.stringify(cleanWorkspace())); }
  catch { toast("Could not save to browser storage. Export your conversation to keep it."); }
}
async function streamBranch(branch, messages, recipe, steering, settings) {
  const id = crypto.randomUUID(); activeRequest = id;
  const response = await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id, messages, recipe, steering, thinking: settings.thinking, max_context: settings.maxContext})});
  if (!response.ok) { const data = await response.json(); throw new Error(data.error || "Chat request failed."); }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = "", finished = false;
  function consume(line) {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.type === "phase") $("phase").textContent = event.message;
    if (event.type === "calibration") { branch.calibration = event; $("phase").textContent = steering ? "Steering every token · generating locally…" : "Generating baseline locally…"; }
    if (event.type === "token") { branch.text += event.text; updateBranch(branch); scrollChat(); }
    if (event.type === "error") throw new Error(event.error);
    if (event.type === "done") { finished = true; branch.meta = event; branch.done = true; branch.cancelled = event.cancelled; updateBranch(branch); }
  }
  try {
    while (true) {
      const {value, done} = await reader.read();
      if (done) { buffer += decoder.decode(); if (buffer.trim()) consume(buffer); break; }
      buffer += decoder.decode(value, {stream: true});
      let end;
      while ((end = buffer.indexOf("\n")) >= 0) { const line = buffer.slice(0, end); buffer = buffer.slice(end + 1); consume(line); }
    }
    if (!finished) throw new Error("The stream disconnected before completion. Your partial response has been kept.");
  } finally { reader.releaseLock(); activeRequest = null; }
}
async function sendMessage(event) {
  event.preventDefault();
  if (busy || preparing) return;
  const prompt = $("prompt").value.trim(); if (!prompt) return;
  if (status?.runtime.state !== "ready") { openDialog("modelsDialog"); return; }
  showError("chatError", null);
  let recipe;
  preparing = true; $("send").disabled = true;
  try { recipe = await api("/api/recipes/import", {recipe: state.recipe}); }
  catch (error) { showError("chatError", error); return; }
  finally { preparing = false; renderModels(); }
  const prior = [...state.turns];
  const histories = {baseline: historyFor("baseline", prior), steered: historyFor("steered", prior)};
  const settings = {thinking: state.thinking, maxContext: state.maxContext};
  const enabled = state.steering && recipe.neurons.some(x => x.enabled);
  if (enabled && recipe.model !== status.runtime.model.repo) { showError("chatError", `This recipe targets ${recipe.model}. Load that model or change the Recipe model field to the model you intend to tweak.`); return; }
  const turn = {prompt, steering: enabled, recipe: structuredClone(recipe), model: status.runtime.model.repo, settings, created: new Date().toISOString(), steered: {text: "", tweaks: enabled ? recipe.neurons.filter(x => x.enabled).length : 0}};
  if (state.compare) turn.baseline = {text: "", tweaks: 0};
  state.turns.push(turn); $("welcome").hidden = true; appendTurn(turn);
  $("prompt").value = ""; busy = true; stopping = false; $("send").hidden = true; $("stop").hidden = false;
  $("newChat").disabled = true; $("compare").disabled = true;
  renderModels(); scrollChat();
  try {
    if (turn.baseline) {
      $("phase").textContent = "Running unsteered baseline first…";
      try { await streamBranch(turn.baseline, [...histories.baseline, {role: "user", content: prompt}], recipe, false, settings); }
      catch (error) { turn.baseline.error = error.message; updateBranch(turn.baseline); throw error; }
    }
    if (!stopping) {
      $("phase").textContent = enabled ? "Preparing neuron steering…" : "Generating locally…";
      try { await streamBranch(turn.steered, [...histories.steered, {role: "user", content: prompt}], recipe, enabled, settings); }
      catch (error) { turn.steered.error = error.message; updateBranch(turn.steered); throw error; }
    } else { turn.steered.done = true; turn.steered.cancelled = true; updateBranch(turn.steered); }
  } catch (error) {
    if (turn.baseline?.error && !turn.steered.text) { turn.steered.error = "Not run because the baseline failed."; updateBranch(turn.steered); }
    showError("chatError", error);
  } finally {
    busy = false; activeRequest = null; stopping = false;
    $("phase").textContent = ""; $("send").hidden = false; $("stop").hidden = true; $("stop").disabled = false; $("stop").textContent = "■ Stop";
    $("newChat").disabled = false; $("compare").disabled = false;
    saveImmediately(); await refresh(); $("prompt").focus();
  }
}

for (const id of ["openModels", "modelPicker", "welcomeModels"]) $(id).addEventListener("click", () => { openDialog("modelsDialog"); refresh(); });
for (const id of ["openImport", "welcomeRecipe"]) $(id).addEventListener("click", () => openDialog("importDialog"));
document.querySelectorAll(".close-dialog").forEach(button => button.addEventListener("click", () => button.closest("dialog").close()));
document.querySelectorAll("dialog").forEach(dialog => dialog.addEventListener("click", event => { if (event.target === dialog) { const rect = dialog.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close(); } }));
$("recipeName").addEventListener("input", () => { state.recipe.name = $("recipeName").value; persist(); });
$("recipeModel").addEventListener("input", () => { state.recipe.model = $("recipeModel").value.trim(); persist(); });
$("steeringEnabled").addEventListener("change", () => { state.steering = $("steeringEnabled").checked; updateSummary(); persist(); });
$("addTweak").addEventListener("click", addTweak);
for (const [key, id] of Object.entries(samplingFields)) $(id).addEventListener("input", () => { state.recipe.sampling[key] = $(id).value === "" ? null : Number($(id).value); persist(); });
for (const [id, key] of [["compare", "compare"], ["thinking", "thinking"]]) $(id).addEventListener("change", () => { state[key] = $(id).checked; persist(); });
$("maxContext").addEventListener("input", () => { state.maxContext = Number($("maxContext").value); persist(); });
$("systemPrompt").addEventListener("input", () => { state.systemPrompt = $("systemPrompt").value; persist(); });
$("exportRecipe").addEventListener("click", () => downloadJSON(state.recipe, `${slug(state.recipe.name)}.json`));
$("saveRecipe").addEventListener("click", async () => {
  try {
    const recipe = await api("/api/recipes/import", {recipe: state.recipe});
    const i = state.saved.findIndex(x => x.name === recipe.name);
    if (i >= 0) state.saved[i] = recipe; else state.saved.push(recipe);
    renderSaved(); persist(); toast("Recipe saved in this browser.");
  } catch (error) { toast(error.message); }
});
$("savedRecipes").addEventListener("change", () => { const i = $("savedRecipes").value; if (i !== "") setRecipe(state.saved[Number(i)]); });
$("inspectForm").addEventListener("submit", event => { event.preventDefault(); inspectRepo(); });
$("inspectQwen").addEventListener("click", () => { $("repoId").value = "Qwen/Qwen3.5-4B"; $("revision").value = "main"; inspectRepo(); });
$("downloadButton").addEventListener("click", startDownload);
for (const [id, mode] of [["selectAll", "all"], ["selectNone", "none"], ["selectRecommended", "recommended"]]) $(id).addEventListener("click", () => {
  if (!inspected) return;
  const safe = inspected.files.some(x => x.name.endsWith(".safetensors"));
  for (const file of inspected.files) file.selected = mode === "all" || (mode === "recommended" && (/\.(json|safetensors|model|tiktoken|txt|jinja)$/.test(file.name) || (!safe && file.name.endsWith(".bin"))));
  renderFileList();
});
$("registerForm").addEventListener("submit", async event => {
  event.preventDefault(); showError("modelError", null);
  try { await api("/api/models/register", {path: $("localPath").value.trim()}); $("localPath").value = ""; await refresh(); }
  catch (error) { showError("modelError", error); }
});
$("unloadModel").addEventListener("click", async () => { try { await api("/api/models/unload", {}); await refresh(); toast("Model unloaded. Memory released."); } catch (error) { showError("modelError", error); } });
for (const [id, mode] of [["linkTab", "link"], ["jsonTab", "json"]]) $(id).addEventListener("click", () => {
  importMode = mode; $("linkImport").hidden = mode !== "link"; $("jsonImport").hidden = mode !== "json";
  $("linkTab").classList.toggle("selected", mode === "link"); $("jsonTab").classList.toggle("selected", mode === "json"); showError("importError", null);
});
$("importForm").addEventListener("submit", async event => {
  event.preventDefault(); $("importButton").disabled = true; $("importButton").textContent = "Importing…"; showError("importError", null);
  try {
    const body = importMode === "link" ? {url: $("recipeUrl").value.trim()} : {recipe: JSON.parse($("recipeJson").value)};
    const recipe = await api("/api/recipes/import", body); setRecipe(recipe); $("importDialog").close(); toast(`Imported ${recipe.name} · ${recipe.neurons.length} neurons`);
  } catch (error) { showError("importError", error); }
  finally { $("importButton").disabled = false; $("importButton").textContent = "Import into rack ↙"; }
});
$("chatForm").addEventListener("submit", sendMessage);
$("prompt").addEventListener("keydown", event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("chatForm").requestSubmit(); } });
$("stop").addEventListener("click", async () => {
  stopping = true; $("stop").disabled = true; $("stop").textContent = "Stopping…";
  if (activeRequest) try { await api("/api/chat/cancel", {id: activeRequest}); } catch (error) { showError("chatError", error); $("stop").disabled = false; }
});
$("newChat").addEventListener("click", () => { if (busy) return; state.turns = []; renderConversation(); showError("chatError", null); persist(); $("prompt").focus(); });
$("exportChat").addEventListener("click", () => downloadJSON({app: "Ntweak", exported: new Date().toISOString(), ...cleanWorkspace()}, "ntweak-conversation.json"));
document.querySelectorAll("[data-prompt]").forEach(button => button.addEventListener("click", () => { $("prompt").value = button.dataset.prompt; $("prompt").focus(); }));
document.addEventListener("keydown", event => { if (event.key.toLowerCase() === "m" && !event.metaKey && !event.ctrlKey && !event.altKey && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName) && !document.querySelector("dialog[open]")) openDialog("modelsDialog"); });
window.addEventListener("beforeunload", () => { saveImmediately(); if (activeRequest) fetch("/api/chat/cancel", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({id: activeRequest}), keepalive: true}).catch(() => {}); });

// A previous tab may have closed while streaming. Keep the text, mark that response interrupted.
for (const turn of state.turns) for (const key of ["baseline", "steered"]) if (turn[key] && !turn[key].done && !turn[key].error) { turn[key].done = true; turn[key].cancelled = true; }
syncRecipeFields(); renderConversation(); refresh();
setInterval(() => { if (!document.hidden) refresh(); }, 1600);

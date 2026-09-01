const state = {
  runs: [],
  selectedRun: null,
  selectedRecord: null,
  records: [],
  includeHidden: false,
  currentRun: null,
  currentRecord: null,
  editingAnnotationId: null,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("'", "&#39;");
}

function renderInlineMarkdown(value, assets = []) {
  let text = String(value ?? "");
  const assetByPath = new Map();
  for (const asset of assets || []) {
    const relative = String(asset.relative_path || "");
    const fileName = relative.split("/").pop();
    if (relative) assetByPath.set(relative, asset.url);
    if (fileName) assetByPath.set(fileName, asset.url);
    const bundleImagePath = relative.split("/images/").pop();
    if (bundleImagePath) assetByPath.set(`images/${bundleImagePath}`, asset.url);
  }

  text = text.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (match, alt, src) => {
    const resolved = assetByPath.get(src);
    if (!resolved) return match;
    return `\n<img class="asset-image" src="${escapeAttribute(resolved)}" alt="${escapeAttribute(alt || "benchmark image")}" loading="lazy">\n`;
  });

  return text
    .split(/(<img class="asset-image"[^>]*>)/g)
    .map((part) => part.startsWith("<img ") ? part : escapeHtml(part))
    .join("");
}

function badge(text, cls = "") {
  return `<span class="pill ${cls}">${escapeHtml(text || "unknown")}</span>`;
}

function compactScoreLabel(value) {
  return String(value || "pending").replace(/^Verifier\s+/i, "");
}

function renderRecordScoreBadges(groupResults = []) {
  const labels = {
    single_llm_skills_on: "on",
    single_llm_skills_off: "off",
  };
  const preferred = groupResults
    .filter((item) => labels[item.group_id])
    .map((item) => `<span class="pill mini-score ${escapeAttribute(item.outcome || "")}" title="${escapeAttribute(item.group_id)}">${labels[item.group_id]} ${escapeHtml(compactScoreLabel(item.score_label))}</span>`);
  if (preferred.length) {
    return `<span class="score-badge-strip">${preferred.join("")}</span>`;
  }
  const fallback = groupResults[0] || {};
  return badge(fallback.score_label || "pending", fallback.outcome || "");
}

function formatRunScore(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  return number.toFixed(2);
}

function renderRunScoreComparison(run) {
  const groups = run.summary?.groups || {};
  const onScore = formatRunScore(groups.single_llm_skills_on?.avg_normalized_score);
  const offScore = formatRunScore(groups.single_llm_skills_off?.avg_normalized_score);
  const parts = [];
  if (onScore !== null) parts.push(`<span>on ${escapeHtml(onScore)}</span>`);
  if (offScore !== null) parts.push(`<span>off ${escapeHtml(offScore)}</span>`);
  if (onScore !== null && offScore !== null) {
    const delta = Number(onScore) - Number(offScore);
    const deltaText = delta === 0 ? "0.00" : `${delta > 0 ? "+" : ""}${delta.toFixed(2)}`;
    const deltaClass = delta > 0 ? "positive" : delta < 0 ? "negative" : "neutral";
    parts.push(`<span class="delta ${deltaClass}">Δ ${escapeHtml(deltaText)}</span>`);
  }
  return parts.length ? `<p class="muted run-score-line">${parts.join(" · ")}</p>` : "";
}

function pct(progress) {
  const total = Number(progress?.total || 0);
  const completed = Number(progress?.completed || 0);
  if (!total) return 0;
  return Math.max(0, Math.min(100, Math.round((completed / total) * 100)));
}

function renderGroupDuration(diagnostics = {}) {
  const agentDuration = diagnostics.agent_duration_seconds;
  if (typeof agentDuration === "number" && Number.isFinite(agentDuration)) {
    return `answer ${Math.round(agentDuration)}s`;
  }
  return `elapsed ${Math.round(Number(diagnostics.elapsed_seconds) || 0)}s`;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function currentRunSummary() {
  return state.runs.find((run) => run.run_id === state.selectedRun) || state.currentRun || null;
}

function optionMarkup(values, selected) {
  return [`<option value="">全部</option>`]
    .concat(values.map((value) => `<option value="${escapeAttribute(value)}" ${value === selected ? "selected" : ""}>${escapeHtml(value)}</option>`))
    .join("");
}

async function loadRuns() {
  state.runs = await api(`/api/runs${state.includeHidden ? "?include_hidden=true" : ""}`);
  renderFilterOptions();
  renderRuns();
  const stillVisible = state.runs.some((run) => run.run_id === state.selectedRun);
  if (!stillVisible) {
    state.selectedRun = null;
    state.selectedRecord = null;
  }
  if (!state.selectedRun && state.runs.length) {
    await selectRun(state.runs[0].run_id);
  }
}

async function refreshRuns() {
  const button = $("refresh-button");
  button.classList.add("is-refreshing");
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  try {
    await loadRuns();
  } finally {
    button.classList.remove("is-refreshing");
    button.disabled = false;
    button.removeAttribute("aria-busy");
  }
}

function renderFilterOptions() {
  const datasetFilter = $("dataset-filter");
  const subsetFilter = $("subset-filter");
  const selectedDataset = datasetFilter.value;
  const selectedSubset = subsetFilter.value;
  const datasets = Array.from(new Set(state.runs.flatMap((run) => run.datasets || []))).filter(Boolean).sort();
  const subsets = Array.from(new Set(state.runs.flatMap((run) => run.subsets || []))).filter(Boolean).sort();
  datasetFilter.innerHTML = optionMarkup(datasets, selectedDataset);
  subsetFilter.innerHTML = optionMarkup(subsets, selectedSubset);
}

function renderRuns() {
  const query = $("search-input").value.toLowerCase();
  const status = $("status-filter").value;
  const dataset = $("dataset-filter").value;
  const subset = $("subset-filter").value;
  const rows = state.runs
    .filter((run) => !status || run.status === status)
    .filter((run) => !dataset || (run.datasets || []).includes(dataset))
    .filter((run) => !subset || (run.subsets || []).includes(subset))
    .filter((run) => {
      const text = `${run.run_id} ${run.alias || ""} ${(run.dataset_files || []).join(" ")} ${(run.datasets || []).join(" ")} ${(run.subsets || []).join(" ")}`.toLowerCase();
      return !query || text.includes(query);
    })
    .map((run) => {
      const active = state.selectedRun === run.run_id ? "active" : "";
      return `<button class="run-card ${active}" data-run="${escapeHtml(run.run_id)}">
        <div class="row-top">
          <span class="id-text">${run.favorite ? "★ " : ""}${escapeHtml(run.alias || run.run_id)}</span>
          ${badge(run.status, run.status)}
        </div>
        <p class="muted">${escapeHtml(run.run_id)}</p>
        <div class="bar" aria-label="进度"><div style="width:${pct(run.progress)}%"></div></div>
        <p class="muted">${run.progress?.completed || 0}/${run.progress?.total || 0} · ${run.group_count || 0} groups</p>
        ${renderRunScoreComparison(run)}
        <p class="muted">${escapeHtml((run.datasets || []).join(", ") || "unknown dataset")}</p>
      </button>`;
    })
    .join("");
  $("run-list").innerHTML = rows || `<p class="muted" style="padding:14px">没有匹配的 run</p>`;
  document.querySelectorAll(".run-card").forEach((el) => {
    el.addEventListener("click", () => selectRun(el.dataset.run));
  });
}

async function selectRun(runId) {
  state.selectedRun = runId;
  state.selectedRecord = null;
  const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
  state.currentRun = run;
  state.records = await api(`/api/runs/${encodeURIComponent(runId)}/records`);
  $("run-title").textContent = run.alias || runId;
  $("run-subtitle").textContent = `${run.payload?.records || state.records.length} records · ${run.progress?.status || "unknown"}`;
  $("favorite-run").textContent = run.favorite ? "★" : "☆";
  $("hide-run").textContent = run.hidden ? "↩" : "⌫";
  $("hide-run").title = run.hidden ? "恢复 run" : "隐藏 run";
  renderRuns();
  renderProgress(run.progress);
  renderRecords();
  if (state.records.length) await selectRecord(state.records[0].record_id);
}

function renderProgress(progress) {
  const percent = pct(progress);
  const current = Object.entries(progress?.groups || {})
    .map(([group, item]) => item.current_record_id ? `${group}: ${item.current_record_id}` : "")
    .filter(Boolean)
    .join(" · ");
  $("progress-strip").innerHTML = `
    <div class="row-top">
      <span class="muted">${escapeHtml(progress?.status || "unknown")}</span>
      <span class="muted">${progress?.completed || 0}/${progress?.total || 0}</span>
    </div>
    <div class="bar" style="margin:8px 0"><div style="width:${percent}%"></div></div>
    <p class="muted">${escapeHtml(current || "当前没有正在执行的题目")}</p>
  `;
}

function renderRecords() {
  const query = $("search-input").value.toLowerCase();
  const rows = state.records
    .filter((record) => {
      const text = `${record.record_id} ${record.dataset} ${record.subset} ${record.eval_kind}`.toLowerCase();
      return !query || text.includes(query);
    })
    .map((record) => {
      const active = state.selectedRecord === record.record_id ? "active" : "";
      return `<button class="record-row ${active}" data-record="${escapeHtml(record.record_id)}">
        <div class="row-top">
          <span class="id-text">${escapeHtml(record.record_id)}</span>
          ${renderRecordScoreBadges(record.group_results || [])}
        </div>
        <p class="muted">${escapeHtml(record.dataset)} · ${escapeHtml(record.subset)}</p>
        <p class="muted">${escapeHtml(record.eval_kind)} · notes ${record.annotation_count || 0}</p>
      </button>`;
    })
    .join("");
  $("record-list").innerHTML = rows || `<p class="muted" style="padding:14px">没有题目</p>`;
  document.querySelectorAll(".record-row").forEach((el) => {
    el.addEventListener("click", () => selectRecord(el.dataset.record));
  });
}

async function selectRecord(recordId) {
  state.selectedRecord = recordId;
  const record = await api(`/api/runs/${encodeURIComponent(state.selectedRun)}/records/${encodeURIComponent(recordId)}`);
  state.currentRecord = record;
  $("record-title").textContent = record.record_id;
  $("record-subtitle").textContent = `${record.dataset} · ${record.subset} · ${record.eval_kind}`;
  renderRecords();
  renderQuestion(record);
  renderReference(record);
  renderAnnotations(record);
  renderGroups(record);
}

function renderQuestion(record) {
  const text = renderInlineMarkdown(record.question_markdown || record.prompt || "", record.assets || []);
  $("question-content").innerHTML = text || `<span class="muted">无题目内容</span>`;
}

function renderReference(record) {
  const ref = record.reference || {};
  const checkpoints = (ref.checkpoints || [])
    .map((item) => `<li>${escapeHtml(item.text || item.rationale || JSON.stringify(item))}</li>`)
    .join("");
  $("reference-content").innerHTML = `
    <p><strong>标准答案：</strong>${escapeHtml(ref.answer || "未提供")}</p>
    <p style="margin-top:8px"><strong>参考解题路径：</strong>${escapeHtml(ref.reasoning || "未提供")}</p>
    ${checkpoints ? `<ul>${checkpoints}</ul>` : ""}
  `;
}

function renderGroups(record) {
  $("group-results").innerHTML = (record.groups || [])
    .map((group) => {
      const evalPayload = group.evaluation || {};
      const verifier = group.verifier;
      const isolation = group.workspace_isolation || {};
      const executionError = group.diagnostics?.execution_error || {};
      const primaryError = executionError.primary_error || {};
      const observedErrors = executionError.observed_errors || [];
      const findings = (isolation.findings || []).map((finding) => `
        <li>${escapeHtml(finding.access_mode || "unknown")} / ${escapeHtml(finding.operation_outcome || "unknown")} · ${escapeHtml(finding.policy_id || "")} · ${escapeHtml(finding.resolved_path || "")}</li>
      `).join("");
      const observedErrorItems = observedErrors.map((error) => `
        <li>${escapeHtml(error.source || "unknown")}:${escapeHtml(error.line_number ?? "-")} · ${escapeHtml(error.raw || error.message || "unknown error")}</li>
      `).join("");
      return `<section class="group-card">
        <div class="row-top">
          <span class="id-text">${escapeHtml(group.group_id)}</span>
          ${badge(group.score_label, group.outcome)}
        </div>
        <div class="metrics">
          ${badge(`score ${evalPayload.normalized_score ?? evalPayload.score ?? "-"}`)}
          ${badge(renderGroupDuration(group.diagnostics))}
          ${badge(group.status_axes?.answer_availability || "answer")}
          ${group.status_axes?.degraded_execution ? badge("degraded", "running") : ""}
          ${isolation.adjudication ? badge(isolation.adjudication, isolation.adjudication === "non_evaluable" ? "failed" : "running") : ""}
          ${isolation.boundary_status ? badge(`boundary ${isolation.boundary_status}`) : ""}
          ${isolation.contamination_status ? badge(`contamination ${isolation.contamination_status}`, isolation.contamination_status === "confirmed" ? "failed" : "") : ""}
          ${isolation.legacy_schema ? badge("legacy isolation schema") : ""}
        </div>
        ${verifier ? `<p class="muted">Verifier: ${escapeHtml(verifier.status || "")} · ${escapeHtml(verifier.canonical_smiles || "")}</p>` : ""}
        <div class="answer-block">${escapeHtml(group.answer_text || "无答案")}</div>
        <div class="diag-grid">
          <span>OpenClaw tools: ${escapeHtml(group.diagnostics?.openclaw_tool_call_count ?? "-")}</span>
          ${group.skills_enabled ? `<span>Skill calls: ${escapeHtml(group.diagnostics?.skill_tool_call_count ?? "-")}</span>` : ""}
          ${group.skills_enabled ? `<span>Skill failures: ${escapeHtml(group.diagnostics?.skill_tool_failure_count ?? "-")}</span>` : ""}
          ${!group.skills_enabled ? `<span>Exec calls: ${escapeHtml(group.diagnostics?.exec_tool_call_count ?? "-")}</span>` : ""}
          ${!group.skills_enabled ? `<span>Exec failures: ${escapeHtml(group.diagnostics?.exec_tool_failure_count ?? "-")}</span>` : ""}
          <span>Recovery: ${escapeHtml(group.status_axes?.recovery_mode || "none")}</span>
        </div>
        ${executionError.code ? `<details>
          <summary>Execution error · ${escapeHtml(executionError.code)}${primaryError.status_code ? ` · HTTP ${escapeHtml(primaryError.status_code)}` : ""}</summary>
          <p class="muted">${escapeHtml(executionError.message || primaryError.message || "Unknown execution error")}</p>
          ${observedErrorItems ? `<ul>${observedErrorItems}</ul>` : ""}
        </details>` : ""}
        ${findings ? `<details><summary>Workspace boundary findings</summary><ul>${findings}</ul></details>` : ""}
      </section>`;
    })
    .join("");
}

function renderAnnotations(record) {
  const annotations = record.annotations || [];
  $("annotation-list").innerHTML = annotations.length
    ? annotations.map((annotation) => `<section class="annotation-card">
        <div class="row-top">
          <span class="id-text">${escapeHtml(annotation.status || "note")}</span>
          <span class="annotation-actions">
            <button class="mini-button" title="编辑备注" aria-label="编辑备注" data-action="edit-annotation" data-id="${annotation.id}">✎</button>
            <button class="mini-button" title="删除备注" aria-label="删除备注" data-action="delete-annotation" data-id="${annotation.id}">⌫</button>
          </span>
        </div>
        <p>${escapeHtml(annotation.note || "无备注内容")}</p>
        <p class="muted">${escapeHtml((annotation.tags || []).join(", "))} ${annotation.manual_verdict ? `· ${escapeHtml(annotation.manual_verdict)}` : ""}</p>
      </section>`).join("")
    : `<p class="muted">暂无复核备注</p>`;
  document.querySelectorAll("[data-action='edit-annotation']").forEach((el) => {
    el.addEventListener("click", () => openAnnotationDialog(Number(el.dataset.id)));
  });
  document.querySelectorAll("[data-action='delete-annotation']").forEach((el) => {
    el.addEventListener("click", () => deleteAnnotation(Number(el.dataset.id)));
  });
}

async function patchRunMetadata(updates) {
  if (!state.selectedRun) return;
  await api(`/api/runs/${encodeURIComponent(state.selectedRun)}`, {
    method: "PATCH",
    body: JSON.stringify(updates),
  });
  await loadRuns();
  if (state.selectedRun) await selectRun(state.selectedRun);
}

async function refreshProgress() {
  if (!state.selectedRun) return;
  const progress = await api(`/api/runs/${encodeURIComponent(state.selectedRun)}/progress`);
  renderProgress(progress);
  state.runs = state.runs.map((run) => run.run_id === state.selectedRun ? { ...run, progress, status: progress.status } : run);
  renderRuns();
}

function clearAnnotationForm() {
  state.editingAnnotationId = null;
  $("annotation-status").value = "";
  $("annotation-tags").value = "";
  $("annotation-verdict").value = "";
  $("annotation-note").value = "";
}

function openAnnotationDialog(annotationId = null) {
  clearAnnotationForm();
  if (annotationId && state.currentRecord) {
    const annotation = (state.currentRecord.annotations || []).find((item) => Number(item.id) === annotationId);
    if (annotation) {
      state.editingAnnotationId = annotationId;
      $("annotation-status").value = annotation.status || "";
      $("annotation-tags").value = (annotation.tags || []).join(", ");
      $("annotation-verdict").value = annotation.manual_verdict || "";
      $("annotation-note").value = annotation.note || "";
    }
  }
  $("annotation-dialog").showModal();
}

async function saveAnnotation() {
  if (!state.selectedRun || !state.selectedRecord) return;
  const payload = {
    note: $("annotation-note").value,
    status: $("annotation-status").value,
    tags: $("annotation-tags").value.split(",").map((item) => item.trim()).filter(Boolean),
    manual_verdict: $("annotation-verdict").value,
  };
  if (state.editingAnnotationId) {
    await api(`/api/annotations/${state.editingAnnotationId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
  } else {
    await api("/api/annotations", {
      method: "POST",
      body: JSON.stringify({
        run_id: state.selectedRun,
        record_id: state.selectedRecord,
        ...payload,
      }),
    });
  }
  clearAnnotationForm();
  await selectRecord(state.selectedRecord);
}

async function deleteAnnotation(annotationId) {
  await api(`/api/annotations/${annotationId}`, { method: "DELETE" });
  await selectRecord(state.selectedRecord);
}

$("refresh-button").addEventListener("click", refreshRuns);
$("status-filter").addEventListener("change", renderRuns);
$("dataset-filter").addEventListener("change", renderRuns);
$("subset-filter").addEventListener("change", renderRuns);
$("search-input").addEventListener("input", () => {
  renderRuns();
  renderRecords();
});
$("show-hidden-button").addEventListener("click", async () => {
  state.includeHidden = !state.includeHidden;
  $("show-hidden-button").textContent = state.includeHidden ? "●" : "◌";
  await loadRuns();
});
$("favorite-run").addEventListener("click", () => {
  const run = currentRunSummary();
  if (run) patchRunMetadata({ favorite: !run.favorite });
});
$("hide-run").addEventListener("click", () => {
  const run = currentRunSummary();
  if (run) patchRunMetadata({ hidden: !run.hidden });
});
$("note-button").addEventListener("click", () => openAnnotationDialog());
$("save-annotation").addEventListener("click", saveAnnotation);
setInterval(refreshProgress, 2000);
loadRuns().catch((err) => {
  $("run-list").innerHTML = `<p class="muted" style="padding:14px">${escapeHtml(err.message)}</p>`;
});

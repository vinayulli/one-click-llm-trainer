// ── State ──
const state = {
  currentStep: 0,
  projectId: null,
  projectName: '',
  uploadedFiles: [],
  datasetStats: null,
  suggestions: [],
  selectedModel: null,
  trainingJobId: null,
  evalJobId: null,
  deployJobId: null,
  statusPollInterval: null,
  trainingLogs: [],
};

const API = '';  // same origin

const STEPS = [
  { id: 'create',   label: 'Create Project' },
  { id: 'upload',   label: 'Upload Docs' },
  { id: 'dataset',  label: 'Generate Dataset' },
  { id: 'model',    label: 'Select Model' },
  { id: 'train',    label: 'Train' },
  { id: 'evaluate', label: 'Evaluate' },
  { id: 'deploy',   label: 'Deploy' },
];

// ── Stage → Step mapping ──
const STAGE_TO_STEP = {
  created: 0,
  documents_uploaded: 1,
  documents_processed: 2,
  dataset_generated: 2,
  model_suggested: 3,
  training: 4,
  training_complete: 4,
  evaluating: 5,
  evaluation_complete: 5,
  deploying: 6,
  deployed: 6,
};

const STAGE_LABELS = {
  created: 'Created',
  documents_uploaded: 'Docs Uploaded',
  documents_processed: 'Docs Processed',
  dataset_generated: 'Dataset Ready',
  model_suggested: 'Model Selected',
  training: 'Training...',
  training_complete: 'Trained',
  evaluating: 'Evaluating...',
  evaluation_complete: 'Evaluated',
  deploying: 'Deploying...',
  deployed: 'Deployed',
};

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
  renderStepper();
  showStep(0);
  setupUploadZone();
  loadExistingProjects();
});

// ── Stepper ──
function renderStepper() {
  const el = document.getElementById('stepper');
  el.innerHTML = STEPS.map((s, i) => `
    <div class="step-indicator ${i === 0 ? 'active' : 'disabled'}" id="step-ind-${i}" onclick="goToStep(${i})">
      <div class="step-number">${i + 1}</div>
      ${s.label}
    </div>
  `).join('');
}

function updateStepper() {
  STEPS.forEach((_, i) => {
    const el = document.getElementById(`step-ind-${i}`);
    el.className = 'step-indicator';
    if (i < state.currentStep) el.classList.add('completed');
    else if (i === state.currentStep) el.classList.add('active');
    else el.classList.add('disabled');
  });
}

function goToStep(i) {
  if (i <= state.currentStep) {
    showStep(i);
  }
}

function showStep(i) {
  document.querySelectorAll('.step-section').forEach(s => s.classList.remove('active'));
  const section = document.getElementById(`step-${STEPS[i].id}`);
  if (section) section.classList.add('active');
  state.currentStep = Math.max(state.currentStep, i);
  updateStepper();
}

function nextStep() {
  const next = state.currentStep + 1;
  if (next < STEPS.length) {
    state.currentStep = next;
    showStep(next);
  }
}

// ── Helpers ──
function showAlert(container, type, message) {
  const icons = { success: '&#10003;', error: '&#10007;', warning: '&#9888;', info: '&#8505;' };
  const el = document.getElementById(container);
  if (el) {
    el.innerHTML = `<div class="alert alert-${type}">${icons[type] || ''} ${message}</div>`;
  }
}

function clearAlert(container) {
  const el = document.getElementById(container);
  if (el) el.innerHTML = '';
}

async function apiCall(method, path, body = null, isFormData = false) {
  const opts = { method };
  if (body && !isFormData) {
    opts.headers = { 'Content-Type': 'application/json' };
    opts.body = JSON.stringify(body);
  } else if (body && isFormData) {
    opts.body = body;
  }
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'API request failed');
  }
  return res.json();
}

function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

// ── Step 1: Create Project ──
async function createProject() {
  const name = document.getElementById('project-name').value.trim();
  const desc = document.getElementById('project-desc').value.trim();
  if (!name) return showAlert('create-alert', 'error', 'Project name is required');

  const btn = document.getElementById('btn-create');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Creating...';

  try {
    const project = await apiCall('POST', '/api/projects', { name, description: desc });
    state.projectId = project.id;
    state.projectName = project.name;
    showAlert('create-alert', 'success', `Project created! ID: <strong>${project.id}</strong>`);
    setTimeout(() => nextStep(), 800);
  } catch (e) {
    showAlert('create-alert', 'error', e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Create Project';
  }
}

// ── Existing Projects ──
async function loadExistingProjects() {
  try {
    const projects = await apiCall('GET', '/api/projects');
    if (!projects.length) return;

    const card = document.getElementById('existing-projects-card');
    card.style.display = 'block';

    const list = document.getElementById('existing-projects-list');
    list.innerHTML = `<div class="project-list">
      ${projects.map(p => {
        const stage = p.stage || 'created';
        const date = new Date(p.created_at).toLocaleDateString();
        return `
          <div class="project-card" onclick="resumeProject('${p.id}')">
            <div class="project-card-left">
              <div class="project-card-name">${esc(p.name)}</div>
              ${p.description ? `<div class="project-card-desc">${esc(p.description)}</div>` : ''}
              <div class="project-card-meta">
                <span>ID: ${p.id}</span>
                <span>Created: ${date}</span>
              </div>
            </div>
            <div class="project-card-right">
              <span class="stage-badge stage-${stage}">${STAGE_LABELS[stage] || stage}</span>
              <span class="project-card-arrow">&rarr;</span>
            </div>
          </div>
        `;
      }).join('')}
    </div>`;
  } catch (e) {
    // No projects or API not ready — silently ignore
  }
}

async function resumeProject(projectId) {
  try {
    const project = await apiCall('GET', `/api/projects/${projectId}`);
    state.projectId = project.id;
    state.projectName = project.name;

    const stage = project.stage || 'created';
    const targetStep = STAGE_TO_STEP[stage] ?? 0;

    // Unlock all steps up to and including the target
    state.currentStep = targetStep;

    showAlert('create-alert', 'success',
      `Resumed project <strong>${esc(project.name)}</strong> (${STAGE_LABELS[stage] || stage})`);

    // Restore step-specific data based on stage
    if (targetStep >= 2) {
      // Dataset was generated — try to load stats + preview
      restoreDatasetStep(projectId);
    }
    if (targetStep >= 3) {
      // Model was suggested — load suggestions
      setTimeout(() => loadModelSuggestions(), 100);
    }
    if (targetStep >= 4) {
      // Training started or completed — restore status
      restoreTrainingStep(projectId, stage);
    }
    if (targetStep >= 5) {
      // Eval started or completed — restore
      restoreEvalStep(projectId, stage);
    }
    if (targetStep >= 6) {
      // Deployment — restore
      restoreDeployStep(projectId, stage);
    }

    // Navigate to the correct step
    showStep(targetStep);
    updateStepper();
  } catch (e) {
    showAlert('create-alert', 'error', `Failed to resume: ${e.message}`);
  }
}

async function restoreDatasetStep(projectId) {
  try {
    const result = await apiCall('GET', `/api/projects/${projectId}/dataset`);
    state.datasetStats = result.stats;

    document.getElementById('dataset-stats').innerHTML = `
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-value">${result.stats.total_examples}</div><div class="stat-label">Total Examples</div></div>
        <div class="stat-card"><div class="stat-value">${result.stats.train_count}</div><div class="stat-label">Train</div></div>
        <div class="stat-card"><div class="stat-value">${result.stats.validation_count}</div><div class="stat-label">Validation</div></div>
        <div class="stat-card"><div class="stat-value">${result.stats.eval_count}</div><div class="stat-label">Eval</div></div>
        <div class="stat-card"><div class="stat-value">${Math.round(result.stats.estimated_tokens).toLocaleString()}</div><div class="stat-label">Est. Tokens</div></div>
      </div>
    `;

    if (result.preview && result.preview.length) {
      document.getElementById('dataset-preview').innerHTML = `
        <h3 style="font-size:0.95rem;margin-bottom:12px;color:var(--text-bright)">Dataset Preview</h3>
        <div style="overflow-x:auto">
          <table class="preview-table">
            <thead><tr><th>Instruction</th><th>Input</th><th>Output</th></tr></thead>
            <tbody>
              ${result.preview.map(r => `<tr>
                <td title="${esc(r.instruction)}">${esc(trunc(r.instruction, 80))}</td>
                <td title="${esc(r.input)}">${esc(trunc(r.input || '—', 60))}</td>
                <td title="${esc(r.output)}">${esc(trunc(r.output, 80))}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    showAlert('dataset-alert', 'success', `Dataset loaded: ${result.stats.total_examples} examples`);
    document.getElementById('btn-next-model').style.display = 'inline-flex';
  } catch (e) {
    // Dataset not yet generated at this step
  }
}

async function restoreTrainingStep(projectId, stage) {
  try {
    const result = await apiCall('GET', `/api/projects/${projectId}/train/status`);
    state.trainingJobId = result.job_id;
    state.selectedModel = result.base_model;
    renderTrainingConfig();

    if (result.status === 'completed' || stage === 'training_complete') {
      showAlert('train-alert', 'success', `Training completed. Model: ${result.base_model}`);
      document.getElementById('btn-train').style.display = 'none';
      document.getElementById('btn-next-eval').style.display = 'inline-flex';
    } else if (result.status === 'running') {
      showAlert('train-alert', 'info', `Training in progress on RunPod. Pod: ${result.runpod_pod_id}`);
      addTrainingLog('info', `Resumed. Model: ${result.base_model}`);
      document.getElementById('training-progress').style.display = 'block';
      document.getElementById('btn-train').style.display = 'none';
      document.getElementById('btn-cancel-train').style.display = 'inline-flex';
      state.statusPollInterval = setInterval(pollTrainingStatus, 8000);
      pollTrainingStatus();
    }
  } catch (e) {
    // No training job yet
  }
}

async function restoreEvalStep(projectId, stage) {
  try {
    const result = await apiCall('GET', `/api/projects/${projectId}/evaluate/results`);
    if (result.status === 'completed' && result.num_examples > 0) {
      renderEvalResults(result);
      showAlert('eval-alert', 'success', 'Evaluation complete');
      document.getElementById('btn-eval').style.display = 'none';
      document.getElementById('btn-next-deploy').style.display = 'inline-flex';
    }
  } catch (e) {
    // Eval not run yet
  }
}

async function restoreDeployStep(projectId, stage) {
  try {
    const result = await apiCall('GET', `/api/projects/${projectId}/deploy/status`);
    if (result.status === 'deployed' && result.endpoint_url) {
      renderDeployment(result);
      showAlert('deploy-alert', 'success', 'Model is deployed and running');
      document.getElementById('btn-deploy').style.display = 'none';
    }
  } catch (e) {
    // Not deployed yet
  }
}

// ── Step 2: Upload Documents ──
function setupUploadZone() {
  const zone = document.getElementById('upload-zone');
  const input = document.getElementById('file-input');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
  });
  input.addEventListener('change', () => handleFiles(input.files));
}

function handleFiles(fileList) {
  for (const f of fileList) {
    const ext = f.name.split('.').pop().toLowerCase();
    if (['pdf', 'docx', 'txt'].includes(ext)) {
      if (!state.uploadedFiles.find(x => x.name === f.name)) {
        state.uploadedFiles.push(f);
      }
    }
  }
  renderFileList();
}

function removeFile(idx) {
  state.uploadedFiles.splice(idx, 1);
  renderFileList();
}

function renderFileList() {
  const el = document.getElementById('file-list');
  if (!state.uploadedFiles.length) { el.innerHTML = ''; return; }
  el.innerHTML = state.uploadedFiles.map((f, i) => `
    <div class="file-item">
      <span class="name">${f.name}</span>
      <span class="size">${formatBytes(f.size)}</span>
      <button class="remove" onclick="removeFile(${i})">&times;</button>
    </div>
  `).join('');
}

async function uploadAndProcess() {
  if (!state.uploadedFiles.length) return showAlert('upload-alert', 'error', 'Please select at least one file');

  const btn = document.getElementById('btn-upload');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Uploading...';

  try {
    const formData = new FormData();
    state.uploadedFiles.forEach(f => formData.append('files', f));
    await apiCall('POST', `/api/projects/${state.projectId}/upload`, formData, true);
    showAlert('upload-alert', 'success', `${state.uploadedFiles.length} file(s) uploaded`);

    btn.innerHTML = '<span class="spinner"></span> Processing chunks...';
    const result = await apiCall('POST', `/api/projects/${state.projectId}/process`);
    showAlert('upload-alert', 'success',
      `Uploaded ${result.files_processed.length} files &rarr; ${result.total_chunks} text chunks`);

    setTimeout(() => nextStep(), 800);
  } catch (e) {
    showAlert('upload-alert', 'error', e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Upload & Process';
  }
}

// ── Step 3: Generate Dataset ──
async function generateDataset() {
  const btn = document.getElementById('btn-generate');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Generating (this may take a few minutes)...';
  clearAlert('dataset-alert');

  try {
    const result = await apiCall('POST', `/api/projects/${state.projectId}/generate-dataset`);
    state.datasetStats = result.stats;

    // Render stats
    document.getElementById('dataset-stats').innerHTML = `
      <div class="stats-grid">
        <div class="stat-card"><div class="stat-value">${result.stats.total_examples}</div><div class="stat-label">Total Examples</div></div>
        <div class="stat-card"><div class="stat-value">${result.stats.train_count}</div><div class="stat-label">Train</div></div>
        <div class="stat-card"><div class="stat-value">${result.stats.validation_count}</div><div class="stat-label">Validation</div></div>
        <div class="stat-card"><div class="stat-value">${result.stats.eval_count}</div><div class="stat-label">Eval</div></div>
        <div class="stat-card"><div class="stat-value">${Math.round(result.stats.estimated_tokens).toLocaleString()}</div><div class="stat-label">Est. Tokens</div></div>
      </div>
    `;

    // Render preview
    if (result.preview && result.preview.length) {
      document.getElementById('dataset-preview').innerHTML = `
        <h3 style="font-size:0.95rem;margin-bottom:12px;color:var(--text-bright)">Dataset Preview</h3>
        <div style="overflow-x:auto">
          <table class="preview-table">
            <thead><tr><th>Instruction</th><th>Input</th><th>Output</th></tr></thead>
            <tbody>
              ${result.preview.map(r => `<tr>
                <td title="${esc(r.instruction)}">${esc(trunc(r.instruction, 80))}</td>
                <td title="${esc(r.input)}">${esc(trunc(r.input || '—', 60))}</td>
                <td title="${esc(r.output)}">${esc(trunc(r.output, 80))}</td>
              </tr>`).join('')}
            </tbody>
          </table>
        </div>
      `;
    }

    showAlert('dataset-alert', 'success', `Generated ${result.stats.total_examples} instruction-tuning examples`);
    document.getElementById('btn-next-model').style.display = 'inline-flex';
  } catch (e) {
    showAlert('dataset-alert', 'error', e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Generate Dataset';
  }
}

function esc(s) { return s ? s.replace(/"/g, '&quot;').replace(/</g, '&lt;') : ''; }
function trunc(s, n) { return s && s.length > n ? s.slice(0, n) + '...' : (s || ''); }

// ── Step 4: Model Selection ──
async function loadModelSuggestions() {
  const container = document.getElementById('model-suggestions');
  container.innerHTML = '<div class="loading-state"><div class="spinner"></div>Analyzing dataset and selecting best models...</div>';

  try {
    const result = await apiCall('GET', `/api/projects/${state.projectId}/suggest-models`);
    state.suggestions = result.suggestions;
    state.selectedModel = result.auto_selected;

    container.innerHTML = `
      <div class="model-grid">
        ${result.suggestions.map((m, i) => `
          <div class="model-card ${m.model_id === state.selectedModel ? 'selected' : ''} ${i === 0 ? 'recommended' : ''}"
               onclick="selectModel('${m.model_id}')" id="model-${i}">
            ${i === 0 ? '<span class="badge badge-recommended">Best Pick</span>' : ''}
            ${m.model_id === state.selectedModel && i !== 0 ? '<span class="badge badge-selected">Selected</span>' : ''}
            <div class="model-name">${m.display_name}</div>
            <div class="model-meta">${m.model_id} &middot; ${m.parameter_count} params &middot; Score: ${m.score}</div>
            <div class="model-stats">
              <span>GPU: <strong>${m.recommended_gpu}</strong></span>
              <span>Est. time: <strong>~${m.estimated_train_time_hours}h</strong></span>
              <span>Est. cost: <strong>~$${m.estimated_cost_usd}</strong></span>
            </div>
            <div class="model-reasoning">${m.reasoning}</div>
          </div>
        `).join('')}
      </div>
    `;
  } catch (e) {
    container.innerHTML = `<div class="alert alert-error">${e.message}</div>`;
  }
}

function selectModel(modelId) {
  state.selectedModel = modelId;
  // Re-render cards
  document.querySelectorAll('.model-card').forEach(card => {
    card.classList.remove('selected');
    const badge = card.querySelector('.badge-selected');
    if (badge) badge.remove();
  });
  const idx = state.suggestions.findIndex(m => m.model_id === modelId);
  if (idx >= 0) {
    const card = document.getElementById(`model-${idx}`);
    card.classList.add('selected');
    card.insertAdjacentHTML('afterbegin', '<span class="badge badge-selected">Selected</span>');
  }
}

function renderTrainingConfig() {
  const selectedModel = state.suggestions.find(m => m.model_id === state.selectedModel);
  const el = document.getElementById('training-config');
  if (!el) return;

  const modelName = selectedModel ? selectedModel.display_name : state.selectedModel || '—';
  const modelParams = selectedModel ? selectedModel.parameter_count : '—';
  const gpu = selectedModel ? selectedModel.recommended_gpu : '—';
  const estTime = selectedModel ? selectedModel.estimated_train_time_hours + 'h' : '—';
  const estCost = selectedModel ? '$' + selectedModel.estimated_cost_usd : '—';

  el.innerHTML = `
    <h3 style="font-size:0.9rem;margin-bottom:12px;color:var(--text-bright)">Training Configuration</h3>
    <div class="stats-grid">
      <div class="stat-card"><div class="stat-value" style="font-size:0.85rem">${modelName}</div><div class="stat-label">Model</div></div>
      <div class="stat-card"><div class="stat-value">${modelParams}</div><div class="stat-label">Parameters</div></div>
      <div class="stat-card"><div class="stat-value">${gpu}</div><div class="stat-label">GPU</div></div>
      <div class="stat-card"><div class="stat-value">0.5</div><div class="stat-label">Epochs</div></div>
    </div>
    <div class="stats-grid" style="margin-top:8px">
      <div class="stat-card"><div class="stat-value">QLoRA 4-bit</div><div class="stat-label">Method</div></div>
      <div class="stat-card"><div class="stat-value">r=16, a=32</div><div class="stat-label">LoRA Config</div></div>
      <div class="stat-card"><div class="stat-value" style="color:var(--success)">${estTime}</div><div class="stat-label">Est. Time</div></div>
      <div class="stat-card"><div class="stat-value" style="color:var(--warning)">${estCost}</div><div class="stat-label">Est. Cost</div></div>
    </div>
  `;
  el.style.display = 'block';
}

// ── Step 5: Training ──
async function startTraining() {
  const btn = document.getElementById('btn-train');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Starting training on RunPod...';
  clearAlert('train-alert');

  renderTrainingConfig();

  try {
    const result = await apiCall('POST', `/api/projects/${state.projectId}/train`, {
      base_model: state.selectedModel,
    });
    state.trainingJobId = result.job_id;

    showAlert('train-alert', 'info', `Training started on RunPod. Pod ID: ${result.runpod_pod_id}`);
    addTrainingLog('info', `Training started. Model: ${state.selectedModel}`);
    addTrainingLog('info', `RunPod Pod: ${result.runpod_pod_id}`);

    document.getElementById('training-progress').style.display = 'block';
    btn.style.display = 'none';
    document.getElementById('btn-cancel-train').style.display = 'inline-flex';

    // Start polling
    state.statusPollInterval = setInterval(pollTrainingStatus, 8000);
    pollTrainingStatus();
  } catch (e) {
    showAlert('train-alert', 'error', e.message);
    btn.disabled = false;
    btn.innerHTML = 'Start Training';
  }
}

async function pollTrainingStatus() {
  try {
    const result = await apiCall('GET', `/api/projects/${state.projectId}/train/status`);
    const p = result.progress || {};

    const phase = p.phase || result.status || 'pending';
    const step = p.step || 0;
    const totalSteps = p.total_steps || 0;
    const loss = p.loss;
    const epoch = p.epoch || 0;
    const gpuUtil = p.gpu_util || 0;
    const gpuTemp = p.gpu_temp || 0;
    const gpuMem = p.gpu_mem || 0;
    const etaFormatted = p.eta_formatted || '—';
    const elapsedFormatted = p.elapsed_formatted || '—';
    const stepsPerSec = p.steps_per_second || 0;
    const pct = p.progress_pct || (totalSteps > 0 ? Math.round(step / totalSteps * 100 * 10) / 10 : 0);

    // Update progress bar
    document.getElementById('train-progress-fill').style.width = `${pct}%`;
    document.getElementById('train-progress-pct').textContent = `${pct}%`;

    // Determine phase label
    const phaseLabels = {
      downloading_dataset: 'Downloading dataset...',
      loading_model: 'Loading base model...',
      training: 'Training',
      adapter_saved: 'Adapter saved',
      merging_model: 'Merging LoRA adapter...',
      uploading_to_hf: 'Uploading to HuggingFace...',
      completed: 'Complete',
      failed: 'Failed',
    };
    const phaseLabel = phaseLabels[phase] || phase;

    // Build stats cards — only show metrics we actually have
    const statsEl = document.getElementById('train-stats');
    let row1 = `
      <div class="stat-card">
        <div class="stat-value" style="font-size:1rem;color:var(--accent)">${phaseLabel}</div>
        <div class="stat-label">Phase</div>
      </div>`;

    if (step > 0 || totalSteps > 0) {
      row1 += `<div class="stat-card">
        <div class="stat-value">${step.toLocaleString()} / ${totalSteps > 0 ? totalSteps.toLocaleString() : '?'}</div>
        <div class="stat-label">Steps</div>
      </div>`;
    }
    if (loss !== undefined && loss !== null && loss !== 0) {
      row1 += `<div class="stat-card">
        <div class="stat-value">${loss}</div>
        <div class="stat-label">Loss</div>
      </div>`;
    }
    if (epoch > 0) {
      row1 += `<div class="stat-card">
        <div class="stat-value">${epoch}</div>
        <div class="stat-label">Epoch</div>
      </div>`;
    }

    let row2 = '';
    if (etaFormatted !== '—') {
      row2 += `<div class="stat-card">
        <div class="stat-value" style="color:var(--success)">${etaFormatted}</div>
        <div class="stat-label">ETA</div>
      </div>`;
    }
    if (elapsedFormatted !== '—') {
      row2 += `<div class="stat-card">
        <div class="stat-value">${elapsedFormatted}</div>
        <div class="stat-label">Elapsed</div>
      </div>`;
    }
    if (stepsPerSec > 0) {
      row2 += `<div class="stat-card">
        <div class="stat-value">${stepsPerSec}/s</div>
        <div class="stat-label">Speed</div>
      </div>`;
    }
    if (gpuUtil > 0) {
      row2 += `<div class="stat-card">
        <div class="stat-value">${gpuUtil}%</div>
        <div class="stat-label">GPU Util</div>
      </div>`;
    }

    statsEl.innerHTML = `<div class="stats-grid">${row1}</div>` +
      (row2 ? `<div class="stats-grid" style="margin-top:12px">${row2}</div>` : '');

    // Log significant events
    if (step > 0 && step % 50 === 0) {
      addTrainingLog('info', `Step ${step}/${totalSteps} | Loss: ${loss} | ETA: ${etaFormatted}`);
    }

    if (result.status === 'completed' || phase === 'completed') {
      clearInterval(state.statusPollInterval);
      addTrainingLog('success', 'Training completed! Model uploaded to HuggingFace.');
      showAlert('train-alert', 'success', 'Training complete! Model has been merged and uploaded to HuggingFace.');
      document.getElementById('btn-cancel-train').style.display = 'none';
      document.getElementById('btn-next-eval').style.display = 'inline-flex';
      document.getElementById('train-progress-fill').style.width = '100%';
      document.getElementById('train-progress-pct').textContent = '100%';
    } else if (result.status === 'failed' || phase === 'failed') {
      clearInterval(state.statusPollInterval);
      const errMsg = p.error || 'Check RunPod logs for details.';
      addTrainingLog('error', `Training failed: ${errMsg}`);
      showAlert('train-alert', 'error', `Training failed: ${errMsg}`);
    }
  } catch (e) {
    addTrainingLog('error', `Status poll error: ${e.message}`);
  }
}

async function cancelTraining() {
  if (!confirm('Cancel the training job? This will terminate the RunPod pod.')) return;
  try {
    await apiCall('POST', `/api/projects/${state.projectId}/train/cancel`);
    clearInterval(state.statusPollInterval);
    showAlert('train-alert', 'warning', 'Training cancelled');
    addTrainingLog('error', 'Training cancelled by user');
    document.getElementById('btn-cancel-train').style.display = 'none';
    document.getElementById('btn-train').style.display = 'inline-flex';
    document.getElementById('btn-train').disabled = false;
    document.getElementById('btn-train').innerHTML = 'Restart Training';
  } catch (e) {
    showAlert('train-alert', 'error', e.message);
  }
}

function addTrainingLog(type, msg) {
  const time = new Date().toLocaleTimeString();
  state.trainingLogs.push({ type, msg, time });
  const el = document.getElementById('training-log');
  el.innerHTML = state.trainingLogs.map(l =>
    `<div class="log-entry ${l.type}">[${l.time}] ${l.msg}</div>`
  ).join('');
  el.scrollTop = el.scrollHeight;
}

// ── Step 6: Evaluation ──
async function startEvaluation() {
  const btn = document.getElementById('btn-eval');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Starting evaluation on RunPod...';
  clearAlert('eval-alert');

  try {
    const result = await apiCall('POST', `/api/projects/${state.projectId}/evaluate`);
    state.evalJobId = result.job_id;
    showAlert('eval-alert', 'info', 'Evaluation started. Comparing base model vs fine-tuned model...');

    document.getElementById('eval-progress').style.display = 'block';

    // Poll for results
    const pollEval = setInterval(async () => {
      try {
        const status = await apiCall('GET', `/api/projects/${state.projectId}/evaluate/status`);
        if (status.status === 'completed') {
          clearInterval(pollEval);
          const results = await apiCall('GET', `/api/projects/${state.projectId}/evaluate/results`);
          renderEvalResults(results);
          showAlert('eval-alert', 'success', 'Evaluation complete!');
          btn.style.display = 'none';
          document.getElementById('btn-next-deploy').style.display = 'inline-flex';
        } else if (status.status === 'failed') {
          clearInterval(pollEval);
          showAlert('eval-alert', 'error', 'Evaluation failed');
          btn.disabled = false;
          btn.innerHTML = 'Retry Evaluation';
        }
      } catch (e) { /* keep polling */ }
    }, 15000);
  } catch (e) {
    showAlert('eval-alert', 'error', e.message);
    btn.disabled = false;
    btn.innerHTML = 'Start Evaluation';
  }
}

function renderEvalResults(results) {
  const el = document.getElementById('eval-results');
  const base = results.base_avg_metrics || {};
  const ft = results.finetuned_avg_metrics || {};

  const metrics = ['exact_match', 'f1', 'llm_judge'];
  const labels = { exact_match: 'Exact Match', f1: 'F1 Score', llm_judge: 'LLM Judge' };

  el.innerHTML = `
    <h3 style="font-size:0.95rem;margin-bottom:16px;color:var(--text-bright)">
      Base Model vs Fine-Tuned Model (${results.num_examples} eval examples)
    </h3>
    <table class="comparison-table">
      <thead><tr><th>Metric</th><th>Base Model</th><th>Fine-Tuned</th><th>Improvement</th></tr></thead>
      <tbody>
        ${metrics.map(m => {
          const bv = base[m] || 0;
          const fv = ft[m] || 0;
          const diff = fv - bv;
          const diffStr = diff > 0 ? `+${(diff * 100).toFixed(1)}%` : `${(diff * 100).toFixed(1)}%`;
          return `<tr>
            <td>${labels[m] || m}</td>
            <td class="${bv > fv ? 'better' : 'worse'}">${(bv * 100).toFixed(1)}%</td>
            <td class="${fv >= bv ? 'better' : 'worse'}">${(fv * 100).toFixed(1)}%</td>
            <td class="${diff >= 0 ? 'better' : 'worse'}">${diffStr}</td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>
    <div class="stats-grid" style="margin-top:20px">
      <div class="stat-card"><div class="stat-value">${results.num_examples}</div><div class="stat-label">Eval Examples</div></div>
      <div class="stat-card"><div class="stat-value">${results.base_model || '—'}</div><div class="stat-label">Base Model</div></div>
    </div>
  `;

  document.getElementById('eval-progress').style.display = 'none';
}

// ── Step 7: Deploy ──
async function deployModel() {
  const btn = document.getElementById('btn-deploy');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Deploying on RunPod with vLLM...';
  clearAlert('deploy-alert');

  try {
    const result = await apiCall('POST', `/api/projects/${state.projectId}/deploy`);
    state.deployJobId = result.job_id || '';

    showAlert('deploy-alert', 'info', 'Deployment started on RunPod. Waiting for vLLM to load the model...');

    // Poll for deployment readiness
    const pollDeploy = setInterval(async () => {
      try {
        const status = await apiCall('GET', `/api/projects/${state.projectId}/deploy/status`);
        if (status.status === 'deployed' && status.endpoint_url) {
          clearInterval(pollDeploy);
          renderDeployment(status);
          showAlert('deploy-alert', 'success', 'Model deployed and ready!');
          btn.style.display = 'none';
        } else if (status.status === 'failed') {
          clearInterval(pollDeploy);
          showAlert('deploy-alert', 'error', 'Deployment failed');
          btn.disabled = false;
          btn.innerHTML = 'Retry Deploy';
        }
      } catch (e) { /* keep polling */ }
    }, 15000);
  } catch (e) {
    showAlert('deploy-alert', 'error', e.message);
    btn.disabled = false;
    btn.innerHTML = 'Deploy Model';
  }
}

async function renderDeployment(status) {
  const el = document.getElementById('deploy-result');

  // Get sample code
  let sampleCode = { python_openai: '', python_requests: '', curl: '', javascript: '' };
  try {
    sampleCode = await apiCall('GET', `/api/projects/${state.projectId}/deploy/sample-code`);
  } catch (e) { /* fallback */ }

  el.innerHTML = `
    <div class="endpoint-display">
      <div class="endpoint-label">Your Model Endpoint</div>
      <div class="endpoint-url">${status.endpoint_url}</div>
    </div>

    <div class="stats-grid">
      <div class="stat-card"><div class="stat-value" style="font-size:0.9rem;word-break:break-all">${status.hf_repo_url}</div><div class="stat-label">HuggingFace Repo</div></div>
      <div class="stat-card"><div class="stat-value" style="font-size:0.9rem">${status.model_name}</div><div class="stat-label">Model</div></div>
    </div>

    <h3 style="font-size:0.95rem;margin:24px 0 12px;color:var(--text-bright)">Sample Code</h3>

    <div class="tabs" id="code-tabs">
      <button class="tab active" onclick="showCodeTab('python-openai')">Python (OpenAI)</button>
      <button class="tab" onclick="showCodeTab('python-requests')">Python (requests)</button>
      <button class="tab" onclick="showCodeTab('curl')">cURL</button>
      <button class="tab" onclick="showCodeTab('javascript')">JavaScript</button>
    </div>

    <div class="tab-content active" id="tc-python-openai">
      <div class="code-block"><button class="copy-btn" onclick="copyCode(this)">Copy</button><pre>${esc(sampleCode.python_openai)}</pre></div>
    </div>
    <div class="tab-content" id="tc-python-requests">
      <div class="code-block"><button class="copy-btn" onclick="copyCode(this)">Copy</button><pre>${esc(sampleCode.python_requests)}</pre></div>
    </div>
    <div class="tab-content" id="tc-curl">
      <div class="code-block"><button class="copy-btn" onclick="copyCode(this)">Copy</button><pre>${esc(sampleCode.curl)}</pre></div>
    </div>
    <div class="tab-content" id="tc-javascript">
      <div class="code-block"><button class="copy-btn" onclick="copyCode(this)">Copy</button><pre>${esc(sampleCode.javascript)}</pre></div>
    </div>

    <div class="btn-group" style="margin-top:24px">
      <a href="${status.hf_repo_url}" target="_blank" class="btn btn-secondary">Open HuggingFace Repo</a>
      <button class="btn btn-secondary" style="color:var(--error);border-color:var(--error)" onclick="stopDeployment()">Stop Deployment</button>
    </div>
  `;
}

function showCodeTab(id) {
  document.querySelectorAll('#code-tabs .tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById(`tc-${id}`).classList.add('active');
}

function copyCode(btn) {
  const code = btn.parentElement.querySelector('pre').textContent;
  navigator.clipboard.writeText(code);
  btn.textContent = 'Copied!';
  setTimeout(() => btn.textContent = 'Copy', 2000);
}

async function stopDeployment() {
  if (!confirm('Stop the deployed model? This will terminate the RunPod pod.')) return;
  try {
    await apiCall('DELETE', `/api/projects/${state.projectId}/deploy`);
    showAlert('deploy-alert', 'warning', 'Deployment stopped');
  } catch (e) {
    showAlert('deploy-alert', 'error', e.message);
  }
}

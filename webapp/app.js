/* ================================================
   CoT Faithfulness Dashboard — App Logic
   ================================================ */

// ---- STATE ----
const state = {
  meta: null,
  aggregate: [],
  judge: [],
  annotations: null,
  mistakes: [],
  // Explorer
  filteredMistakes: [],
  currentMistakeIdx: 0,
  // Annotation explorer
  filteredAnnotations: [],
  currentAnnoIdx: 0,
};

// ---- DATA LOADING ----
async function loadJSON(path) {
  const res = await fetch(path);
  return res.json();
}

async function loadAllData() {
  const [meta, aggregate, judge, annotations, mistakes] = await Promise.all([
    loadJSON('data/meta.json'),
    loadJSON('data/aggregate.json'),
    loadJSON('data/lm_judge.json'),
    loadJSON('data/annotations.json'),
    loadJSON('data/mistakes.json'),
  ]);

  state.meta = meta;
  state.aggregate = aggregate;
  state.judge = judge;
  state.annotations = annotations;
  state.mistakes = mistakes;
  state.filteredMistakes = [...mistakes];
  state.filteredAnnotations = [...(annotations.instances || [])];
}

// ---- INITIALIZATION ----
async function init() {
  await loadAllData();
  renderHero();
  renderJudgeCharts();
  renderAnnotationSummary();
  renderAnnotationBreakdown();
  populateExplorerFilters();
  renderCurrentMistake();
  renderCurrentAnnotation();
  setupEventListeners();
  setupScrollSpy();
}

// ---- HERO ----
function renderHero() {
  const { meta } = state;
  document.getElementById('paper-authors').textContent = meta.authors;
  document.getElementById('arxiv-link').href = meta.arxiv;
  document.getElementById('footer-arxiv').href = meta.arxiv;
}

// ---- JUDGE CHARTS ----
function renderJudgeCharts() {
  const container = document.getElementById('judge-charts');
  container.innerHTML = '';

  // Group by dataset
  const byDataset = {};
  state.judge.forEach(j => {
    if (!byDataset[j.dataset]) byDataset[j.dataset] = [];
    byDataset[j.dataset].push(j);
  });

  for (const [dataset, entries] of Object.entries(byDataset).sort()) {
    const datasetNice = entries[0].dataset_nice;

    entries.sort((a, b) => a.model.localeCompare(b.model));

    entries.forEach(entry => {
      const total = entry.total;
      const agrPct = (entry.agree / total * 100).toFixed(1);
      const disPct = (entry.disagree / total * 100).toFixed(1);
      const uncPct = (entry.unclear / total * 100).toFixed(1);

      const card = document.createElement('div');
      card.className = 'judge-card fade-in';
      card.innerHTML = `
        <div class="judge-card-header">
          <div>
            <div class="judge-card-title">${entry.model_nice}</div>
            <div class="judge-card-subtitle">${datasetNice} · n=${total}</div>
          </div>
          <div class="judge-card-pct" style="color: var(--accent-red)">${disPct}%</div>
        </div>
        <div class="judge-bar-container">
          <div class="judge-bar-agree" style="width: ${agrPct}%"></div>
          <div class="judge-bar-disagree" style="width: ${disPct}%"></div>
          <div class="judge-bar-unclear" style="width: ${uncPct}%"></div>
        </div>
        <div class="judge-legend">
          <span class="legend-item"><span class="legend-dot" style="background:var(--accent-green)"></span> Agree ${agrPct}%</span>
          <span class="legend-item"><span class="legend-dot" style="background:var(--accent-red)"></span> Disagree ${disPct}%</span>
          <span class="legend-item"><span class="legend-dot" style="background:var(--accent-amber)"></span> Unclear ${uncPct}%</span>
        </div>
      `;
      container.appendChild(card);
    });
  }
}

// ---- ANNOTATION SUMMARY ----
function renderAnnotationSummary() {
  const container = document.getElementById('annotation-summary');
  const { summary } = state.annotations;

  const colors = {
    'Not Supportive At All': 'var(--accent-red)',
    'Slightly Supportive': 'var(--accent-amber)',
    'Mostly Supportive': 'var(--accent-blue)',
    'Very Supportive': 'var(--accent-green)',
  };

  container.innerHTML = summary.map(s => `
    <div class="anno-stat-card fade-in">
      <div class="anno-stat-value" style="color: ${colors[s.rating] || 'white'}">${s.percentage}%</div>
      <div class="anno-stat-label">${s.rating}<br>(${s.count} instances)</div>
    </div>
  `).join('');
}

// ---- ANNOTATION BREAKDOWN ----
function renderAnnotationBreakdown() {
  const container = document.getElementById('annotation-breakdown');
  const { breakdown } = state.annotations;

  const ratingKeys = [
    { key: 'very_supportive', label: 'Very Supportive', color: 'var(--accent-green)' },
    { key: 'mostly_supportive', label: 'Mostly Supportive', color: 'var(--accent-blue)' },
    { key: 'slightly_supportive', label: 'Slightly Supportive', color: 'var(--accent-amber)' },
    { key: 'not_supportive_at_all', label: 'Not Supportive', color: 'var(--accent-red)' },
  ];

  container.innerHTML = breakdown.map(b => {
    const bars = ratingKeys.map(r => {
      const count = b[r.key] || 0;
      const pct = b.total > 0 ? (count / b.total * 100) : 0;
      return `
        <div class="anno-bar-row">
          <span class="anno-bar-label">${r.label}</span>
          <div class="anno-bar-track">
            <div class="anno-bar-fill" style="width: ${pct}%; background: ${r.color}"></div>
          </div>
          <span class="anno-bar-count">${count}</span>
        </div>
      `;
    }).join('');

    return `
      <div class="anno-breakdown-card fade-in">
        <div class="anno-breakdown-title">${b.model_nice} · ${b.dataset_nice}</div>
        ${bars}
      </div>
    `;
  }).join('');
}

// ---- EXPLORER ----
function populateExplorerFilters() {
  const datasetSelect = document.getElementById('explorer-dataset');
  const modelSelect = document.getElementById('explorer-model');

  const datasets = [...new Set(state.mistakes.map(m => m.dataset))];
  const models = [...new Set(state.mistakes.map(m => m.model))];

  datasets.forEach(d => {
    const opt = document.createElement('option');
    opt.value = d;
    opt.textContent = state.mistakes.find(m => m.dataset === d)?.dataset_nice || d;
    datasetSelect.appendChild(opt);
  });

  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = state.mistakes.find(i => i.model === m)?.model_nice || m;
    modelSelect.appendChild(opt);
  });
}

function filterMistakes() {
  const dataset = document.getElementById('explorer-dataset').value;
  const model = document.getElementById('explorer-model').value;

  state.filteredMistakes = state.mistakes.filter(m => {
    if (dataset !== 'all' && m.dataset !== dataset) return false;
    if (model !== 'all' && m.model !== model) return false;
    return true;
  });

  state.currentMistakeIdx = 0;
  renderCurrentMistake();
}

function renderCurrentMistake() {
  const container = document.getElementById('explorer-content');
  const items = state.filteredMistakes;

  if (items.length === 0) {
    container.innerHTML = '<div class="explorer-placeholder">No instances match the current filters.</div>';
    document.getElementById('instance-counter').textContent = '0 / 0';
    return;
  }

  const idx = state.currentMistakeIdx;
  const item = items[idx];
  document.getElementById('instance-counter').textContent = `${idx + 1} / ${items.length}`;

  const LETTERS = ['A', 'B', 'C', 'D', 'E'];

  // Render options
  const optionsHtml = item.options.map((opt, i) => {
    return `<div class="option-item">${opt}</div>`;
  }).join('');

  // Render CoT steps with target highlighting
  const stepsHtml = item.segmented_cot.map((step, i) => {
    const isTarget = i === item.step_idx;
    const cls = isTarget ? 'cot-step-target' : 'cot-step-normal';
    return `
      <div class="cot-step ${cls}">
        <span class="cot-step-num">${i + 1}</span>
        <span class="cot-step-text">${escapeHtml(step)}</span>
      </div>
    `;
  }).join('');

  container.innerHTML = `
    <div class="instance-header">
      <span class="instance-tag tag-model">${item.model_nice}</span>
      <span class="instance-tag tag-dataset">${item.dataset_nice}</span>
    </div>

    <div class="instance-question">${escapeHtml(item.question)}</div>

    <div class="instance-options">${optionsHtml}</div>

    <div class="cot-section-title">Chain-of-Thought Steps</div>
    <div class="cot-steps">${stepsHtml}</div>

    <div class="cot-section-title">Mistake Comparison (Step ${item.step_idx + 1})</div>
    <div class="mistake-comparison">
      <div class="mistake-box original-step">
        <div class="mistake-box-label">✓ Original Step</div>
        ${escapeHtml(item.cot_step)}
      </div>
      <div class="mistake-box mistake-step">
        <div class="mistake-box-label">✗ With Mistake Added</div>
        ${escapeHtml(item.mistake_cot_step)}
      </div>
    </div>
  `;
}

// ---- ANNOTATION EXPLORER ----
function filterAnnotations() {
  const rating = document.getElementById('anno-filter-rating').value;
  const flip = document.getElementById('anno-filter-flip').value;

  state.filteredAnnotations = (state.annotations.instances || []).filter(a => {
    if (rating !== 'all' && a.rating !== rating) return false;
    if (flip !== 'all') {
      if (flip === 'true' && !a.flip) return false;
      if (flip === 'false' && a.flip) return false;
    }
    return true;
  });

  state.currentAnnoIdx = 0;
  renderCurrentAnnotation();
}

function renderCurrentAnnotation() {
  const container = document.getElementById('annotation-content');
  const items = state.filteredAnnotations;

  if (items.length === 0) {
    container.innerHTML = '<div class="explorer-placeholder">No annotations match the current filters.</div>';
    document.getElementById('anno-counter').textContent = '0 / 0';
    return;
  }

  const idx = state.currentAnnoIdx;
  const item = items[idx];
  document.getElementById('anno-counter').textContent = `${idx + 1} / ${items.length}`;

  const LETTERS = ['A', 'B', 'C', 'D', 'E'];

  // Options
  const optionsHtml = item.options.map((opt, i) => {
    let cls = 'option-item';
    if (i === item.correct_answer) cls += ' option-correct';
    if (i === item.predicted_answer) cls += ' option-predicted';
    return `<div class="${cls}">${escapeHtml(opt)}${i === item.correct_answer ? ' ✓' : ''}${i === item.predicted_answer && i !== item.correct_answer ? ' (predicted)' : ''}</div>`;
  }).join('');

  // Steps
  const stepsHtml = item.steps.map((step, i) => {
    const isTarget = i === item.target_step_idx;
    const cls = isTarget ? 'cot-step-target' : 'cot-step-normal';
    return `
      <div class="cot-step ${cls}">
        <span class="cot-step-num">${i + 1}</span>
        <span class="cot-step-text">${escapeHtml(step)}</span>
      </div>
    `;
  }).join('');

  const flipTag = item.flip
    ? '<span class="instance-tag tag-flip">⚡ Answer Flipped</span>'
    : '<span class="instance-tag tag-no-flip">→ No Flip</span>';

  const dpClass = item.dp >= 0 ? 'dp-positive' : 'dp-negative';
  const dpSign = item.dp >= 0 ? '+' : '';

  container.innerHTML = `
    <div class="instance-header">
      <span class="instance-tag tag-model">${item.model_nice}</span>
      <span class="instance-tag tag-dataset">${item.dataset_nice}</span>
      <span class="instance-tag tag-rating">📝 ${item.rating}</span>
      ${flipTag}
    </div>

    <div class="instance-question">${escapeHtml(item.question)}</div>

    <div class="instance-options">${optionsHtml}</div>

    <div class="cot-section-title">Chain-of-Thought Steps <span style="font-weight:400; text-transform:none; letter-spacing:0">(target step highlighted)</span></div>
    <div class="cot-steps">${stepsHtml}</div>

    <div class="dp-display">
      <span class="dp-label">Probability shift (Δp):</span>
      <span class="dp-value ${dpClass}">${dpSign}${item.dp}</span>
    </div>
  `;
}

// ---- EVENT LISTENERS ----
function setupEventListeners() {
  // Dashboard view toggle
  document.querySelectorAll('#dashboard-view-toggle .toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#dashboard-view-toggle .toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('.dashboard-view').forEach(v => v.classList.remove('active'));
      const view = btn.dataset.view;
      document.getElementById(view === 'judge' ? 'judge-view' : 'annotations-view').classList.add('active');
    });
  });

  // Explorer filters
  document.getElementById('explorer-dataset').addEventListener('change', filterMistakes);
  document.getElementById('explorer-model').addEventListener('change', filterMistakes);

  // Explorer navigation
  document.getElementById('prev-instance').addEventListener('click', () => {
    if (state.currentMistakeIdx > 0) {
      state.currentMistakeIdx--;
      renderCurrentMistake();
    }
  });
  document.getElementById('next-instance').addEventListener('click', () => {
    if (state.currentMistakeIdx < state.filteredMistakes.length - 1) {
      state.currentMistakeIdx++;
      renderCurrentMistake();
    }
  });

  // Annotation filters
  document.getElementById('anno-filter-rating').addEventListener('change', filterAnnotations);
  document.getElementById('anno-filter-flip').addEventListener('change', filterAnnotations);

  // Annotation navigation
  document.getElementById('prev-anno').addEventListener('click', () => {
    if (state.currentAnnoIdx > 0) {
      state.currentAnnoIdx--;
      renderCurrentAnnotation();
    }
  });
  document.getElementById('next-anno').addEventListener('click', () => {
    if (state.currentAnnoIdx < state.filteredAnnotations.length - 1) {
      state.currentAnnoIdx++;
      renderCurrentAnnotation();
    }
  });
}

// ---- SCROLL SPY ----
function setupScrollSpy() {
  const sections = document.querySelectorAll('section');
  const navLinks = document.querySelectorAll('.nav-link');

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navLinks.forEach(link => link.classList.remove('active'));
        const activeLink = document.querySelector(`.nav-link[href="#${entry.target.id}"]`);
        if (activeLink) activeLink.classList.add('active');
      }
    });
  }, {
    rootMargin: '-50% 0px -50% 0px',
  });

  sections.forEach(section => observer.observe(section));
}

// ---- UTILITIES ----
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// ---- START ----
document.addEventListener('DOMContentLoaded', init);

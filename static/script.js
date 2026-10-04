const form = document.getElementById('form');
const btn = document.getElementById('go');
const errEl = document.getElementById('error');
const result = document.getElementById('result');
const arc = document.getElementById('arc');
const thr = document.getElementById('thr');
const ARC_LEN = 251.33;

const $ = (n) => form.elements[n];
const fmt = new Intl.NumberFormat('en-IN');

// Loan-to-income is derived, so keep it in sync.
function syncPercent() {
  const inc = parseFloat($('person_income').value);
  const amt = parseFloat($('loan_amnt').value);
  const out = $('loan_percent_income');
  if (inc > 0 && amt >= 0) {
    out.value = (amt / inc).toFixed(2);
    document.getElementById('pctLabel').textContent = Math.round((amt / inc) * 100) + '% of income';
  } else {
    out.value = '';
    document.getElementById('pctLabel').textContent = '';
  }
}
['person_income', 'loan_amnt'].forEach((n) => $(n).addEventListener('input', syncPercent));
syncPercent();

// Point on the dial for a probability p (0..1).
function dialPoint(p, r) {
  const a = Math.PI * (1 - p);
  return [100 + r * Math.cos(a), 100 - r * Math.sin(a)];
}

function placeThreshold(t) {
  const [x1, y1] = dialPoint(t, 70);
  const [x2, y2] = dialPoint(t, 92);
  thr.setAttribute('x1', x1); thr.setAttribute('y1', y1);
  thr.setAttribute('x2', x2); thr.setAttribute('y2', y2);
  thr.style.opacity = 1;
}

function countUp(el, to, ms = 1600) {
  const start = performance.now();
  const tick = (now) => {
    const k = Math.min((now - start) / ms, 1);
    const eased = 1 - Math.pow(1 - k, 4);
    el.textContent = (to * eased).toFixed(1);
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function showResult(d) {
  const p = d.default_probability;
  const high = d.default_prediction === 1;

  result.dataset.state = '';
  void result.offsetWidth; // restart CSS animations
  result.dataset.state = high ? 'high' : 'low';

  arc.style.strokeDashoffset = ARC_LEN * (1 - Math.min(Math.max(p, 0), 1));
  placeThreshold(d.threshold);
  countUp(document.getElementById('pct'), p * 100);

  document.getElementById('badge').textContent = high ? 'High risk' : 'Low risk';
  document.getElementById('thrVal').textContent = (d.threshold * 100).toFixed(1) + '%';
  document.getElementById('note').textContent = high
    ? 'The estimated default probability is at or above the cutoff. This application needs closer review.'
    : 'The estimated default probability is below the cutoff. This application looks safe to proceed.';
}

function validate() {
  let ok = true;
  form.querySelectorAll('input[type=number]:not([readonly])').forEach((i) => {
    const bad = i.value === '' || !i.checkValidity();
    i.classList.toggle('bad', bad);
    if (bad) ok = false;
  });
  return ok;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errEl.textContent = '';
  if (!validate()) {
    errEl.textContent = 'Check the highlighted fields and try again.';
    return;
  }

  const f = new FormData(form);
  const int = ['person_age', 'cb_person_cred_hist_length'];
  const num = ['person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income'];
  const body = {};
  for (const [k, v] of f.entries()) {
    body[k] = int.includes(k) ? parseInt(v, 10) : num.includes(k) ? parseFloat(v) : v;
  }

  btn.classList.add('loading');
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error('The server returned ' + res.status + '. Check the input values and try again.');
    showResult(await res.json());
  } catch (err) {
    errEl.textContent = err.message || 'Could not reach the server. Try again in a moment.';
  } finally {
    btn.classList.remove('loading');
  }
});

form.addEventListener('input', (e) => e.target.classList?.remove('bad'));
/**
 * compare.js — EstateIQ Property Comparison Module
 *
 * Responsibilities:
 *  - Wire sliders, steppers for compare cards A, B, C
 *  - Add / Remove Property C
 *  - Read form values and POST to /api/v1/predict (Promise.all)
 *  - Render Chart.js bar chart (gold theme)
 *  - Render summary table and Cheapest / Best Value badges
 */

(function () {
  'use strict';

  /* ─── helpers ──────────────────────────────────────────── */
  const $ = id => document.getElementById(id);
  const fmt  = n => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n);
  const fmtK = n => '$' + (n / 1000).toFixed(0) + 'K';
  const fmtPsf = n => '$' + Number(n).toFixed(2);

  /* Bar colours per property slot */
  const COLORS = {
    A: { bg: 'rgba(201,168,76,0.85)',  border: '#c9a84c' },
    B: { bg: 'rgba(91,156,246,0.85)',  border: '#5b9cf6' },
    C: { bg: 'rgba(76,175,125,0.85)',  border: '#4caf7d' },
  };

  /* ─── Chart instance ────────────────────────────────────── */
  let cmpChartInst = null;

  /* ─── Stepper init (reuse existing pattern from main inline) */
  function initStepper(wrapperId) {
    const wrap = $(wrapperId);
    if (!wrap) return;
    const inp = wrap.querySelector('input');
    wrap.querySelector('.minus').addEventListener('click', function () {
      if (parseInt(inp.value) > parseInt(inp.min)) inp.value = parseInt(inp.value) - 1;
    });
    wrap.querySelector('.plus').addEventListener('click', function () {
      if (parseInt(inp.value) < parseInt(inp.max)) inp.value = parseInt(inp.value) + 1;
    });
  }

  /* ─── Slider live-value display ─────────────────────────── */
  function bindSlider(sliderId, displayId) {
    const slider  = $(sliderId);
    const display = $(displayId);
    if (!slider || !display) return;
    slider.addEventListener('input', function () { display.textContent = slider.value; });
  }

  /* ─── Wire all controls for one card ───────────────────── */
  function initCard(prefix) {
    initStepper('cmp-' + prefix + '-sbeds');
    initStepper('cmp-' + prefix + '-sbaths');
    bindSlider('compare-' + prefix + '-year',      'cmp-' + prefix + '-yearV');
    bindSlider('compare-' + prefix + '-location',  'cmp-' + prefix + '-locV');
    bindSlider('compare-' + prefix + '-condition', 'cmp-' + prefix + '-condV');
  }

  initCard('a');
  initCard('b');
  initCard('c');   // wired even while hidden — safe

  /* ─── Property C show / hide ────────────────────────────── */
  const cardC     = $('cmpCardC');
  const addCBtn   = $('cmpAddC');
  const removeCBtn = $('cmpRemoveC');

  addCBtn.addEventListener('click', function () {
    cardC.style.display = '';
    addCBtn.style.display = 'none';
  });

  removeCBtn.addEventListener('click', function () {
    cardC.style.display = 'none';
    addCBtn.style.display = '';
    // hide results if they were showing C
    hideResults();
  });

  /* ─── Read one property's values from the DOM ───────────── */
  function readProperty(prefix) {
    const area = parseFloat($('compare-' + prefix + '-area').value);
    if (!area || area < 100) return null;   // treat as unfilled
    return {
      label: 'Property ' + prefix.toUpperCase(),
      colorKey: prefix.toUpperCase(),
      payload: {
        area:           area,
        bedrooms:       parseInt($('compare-' + prefix + '-bedrooms').value),
        bathrooms:      parseInt($('compare-' + prefix + '-bathrooms').value),
        floors:         parseInt($('compare-' + prefix + '-floors').value),
        year_built:     parseInt($('compare-' + prefix + '-year').value),
        location_score: parseInt($('compare-' + prefix + '-location').value),
        condition:      parseInt($('compare-' + prefix + '-condition').value),
        garage:         $('compare-' + prefix + '-garage').checked ? 1 : 0,
        pool:           $('compare-' + prefix + '-pool').checked   ? 1 : 0,
        garden:         $('compare-' + prefix + '-garden').checked ? 1 : 0,
      }
    };
  }

  /* ─── POST one property to the predict endpoint ─────────── */
  async function predictOne(prop) {
    const res = await fetch('/api/v1/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(prop.payload)
    });
    const data = await res.json();
    if (!data.success) throw new Error(prop.label + ': ' + (data.errors || ['Prediction failed']).join(', '));
    return {
      label:    prop.label,
      colorKey: prop.colorKey,
      area:     prop.payload.area,
      bedrooms: prop.payload.bedrooms,
      price:    data.price,
      price_psf: data.price_psf,
      market:   data.market_position,
    };
  }

  /* ─── Chart render ──────────────────────────────────────── */
  function renderCompareChart(results) {
    const ctx    = $('cmpChart').getContext('2d');
    const dark   = document.documentElement.getAttribute('data-theme') === 'dark';
    const tc     = dark ? '#9d9a91' : '#5c5650';
    const gc     = dark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)';

    if (cmpChartInst) { cmpChartInst.destroy(); cmpChartInst = null; }

    cmpChartInst = new Chart(ctx, {
      type: 'bar',
      data: {
        labels:   results.map(r => r.label),
        datasets: [{
          label:           'Estimated Price',
          data:            results.map(r => r.price),
          backgroundColor: results.map(r => COLORS[r.colorKey].bg),
          borderColor:     results.map(r => COLORS[r.colorKey].border),
          borderWidth:     2,
          borderRadius:    8,
          borderSkipped:   false,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) { return ' ' + fmt(ctx.raw); }
            }
          }
        },
        scales: {
          x: {
            ticks: { color: tc, font: { size: 11, family: 'DM Sans' } },
            grid:  { color: gc }
          },
          y: {
            ticks: {
              color: tc,
              font:  { size: 10 },
              callback: function (v) { return fmtK(v); }
            },
            grid: { color: gc }
          }
        }
      }
    });
  }

  /* ─── Badge render ──────────────────────────────────────── */
  function renderBadges(results) {
    const cheapest = results.reduce((a, b) => a.price     < b.price     ? a : b);
    const bestVal  = results.reduce((a, b) => a.price_psf < b.price_psf ? a : b);

    $('cmpBadges').innerHTML = `
      <div class="cmp-badge cheapest">
        <span class="cmp-badge-ico">💰</span>
        <div class="cmp-badge-txt">
          <span class="cmp-badge-lbl">Cheapest</span>
          <span class="cmp-badge-val">${cheapest.label} at ${fmt(cheapest.price)}</span>
        </div>
      </div>
      <div class="cmp-badge bestval">
        <span class="cmp-badge-ico">🏆</span>
        <div class="cmp-badge-txt">
          <span class="cmp-badge-lbl">Best Value</span>
          <span class="cmp-badge-val">${bestVal.label} at ${fmtPsf(bestVal.price_psf)}/sq ft</span>
        </div>
      </div>`;
  }

  /* ─── Table render ──────────────────────────────────────── */
  function renderTable(results) {
    const lowestPrice = Math.min(...results.map(r => r.price));
    const rows = results.map(function (r) {
      const isWinner = r.price === lowestPrice;
      const dotClass = 'dot-' + r.colorKey.toLowerCase();
      return `<tr class="${isWinner ? 'winner-row' : ''}">
        <td><span class="cmp-prop-badge"><span class="${dotClass}"></span>${r.label}</span></td>
        <td>${r.area.toLocaleString()}</td>
        <td>${r.bedrooms}</td>
        <td class="price-cell">${fmt(r.price)}</td>
        <td>${fmtPsf(r.price_psf)}</td>
      </tr>`;
    }).join('');
    $('cmpTableBody').innerHTML = rows;
  }

  /* ─── Show / hide results panel ────────────────────────── */
  function showResults() { $('cmpResults').classList.add('visible'); }
  function hideResults() { $('cmpResults').classList.remove('visible'); }

  /* ─── Error display ─────────────────────────────────────── */
  function showCmpError(msg) {
    const el = $('cmpError');
    el.textContent = msg;
    el.style.display = 'block';
    setTimeout(function () { el.style.display = 'none'; }, 5000);
  }

  /* ─── Main compare action ───────────────────────────────── */
  const cmpBtn = $('cmpBtn');

  cmpBtn.addEventListener('click', async function () {
    // Collect active properties
    const propA = readProperty('a');
    const propB = readProperty('b');
    const cVisible = cardC.style.display !== 'none';
    const propC = cVisible ? readProperty('c') : null;

    // Validate — need at least 2
    const active = [propA, propB, propC].filter(Boolean);
    if (active.length < 2) {
      showCmpError('Please fill in at least two properties (Area is required).');
      return;
    }

    // Loading state
    cmpBtn.disabled = true;
    cmpBtn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Comparing...';
    hideResults();

    try {
      // Fire all requests in parallel
      const results = await Promise.all(active.map(predictOne));

      renderBadges(results);
      renderCompareChart(results);
      renderTable(results);
      showResults();
      $('cmpResults').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      showCmpError(err.message || 'One or more predictions failed. Please try again.');
    } finally {
      cmpBtn.disabled = false;
      cmpBtn.innerHTML = '<i class="fa-solid fa-scale-balanced"></i> Compare Prices';
    }
  });

  /* ─── Re-destroy chart on theme change so colours update ── */
  document.getElementById('themeBtn').addEventListener('click', function () {
    if (cmpChartInst) {
      // small delay to let CSS vars update first
      setTimeout(function () {
        const results = Array.from($('cmpTableBody').querySelectorAll('tr')).map(function (tr) {
          const cells = tr.querySelectorAll('td');
          if (cells.length < 5) return null;
          const label    = cells[0].querySelector('.cmp-prop-badge').textContent.trim();
          const colorKey = label.slice(-1);   // 'A', 'B', or 'C'
          const price    = parseFloat(cells[3].textContent.replace(/[^0-9.]/g, ''));
          return { label, colorKey, price };
        }).filter(Boolean);
        if (results.length >= 2) renderCompareChart(results);
      }, 50);
    }
  });

})();

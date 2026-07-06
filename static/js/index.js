/* ===================================================================
   index.js — dynamical switcher + theme toggle.
   =================================================================== */

/* ------------------- TABLE-OF-CONTENTS HIGHLIGHT ------------------
   The TOC links (.switch-btn) jump to sections via their href="#id".
   This highlights whichever section is currently in view.
------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('.switcher-buttons a.switch-btn'));
  if (!links.length || !('IntersectionObserver' in window)) return;

  var byId = {};
  links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });

  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      var link = byId[e.target.id];
      if (!link) return;
      if (e.isIntersecting) {
        links.forEach(function (l) { l.classList.remove('is-active'); });
        link.classList.add('is-active');
      }
    });
  }, { rootMargin: '-40% 0px -55% 0px' });

  links.forEach(function (a) {
    var sec = document.getElementById(a.getAttribute('href').slice(1));
    if (sec) observer.observe(sec);
  });
});

/* ------------------- ATTENTION-MAP BLOCK SLIDER -------------------
   For each .attn-viewer: a slider selects a transformer block and swaps
   the ViT + DINOv2 attention images. Files are named
   vit_block01.png ... dino_block01.png ... (zero-padded to 2 digits).
   Block count comes from data-num-blocks on .attn-viewer.
------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.attn-viewer').forEach(function (viewer) {
    var slider = viewer.querySelector('.attn-slider');
    var output = viewer.querySelector('.attn-value');
    if (!slider) return;

    function pad(b) { return (b < 10 ? '0' : '') + b; }

    // Parse "key:Label, key2:Label2" into [{key,label}, ...].
    function parseModels(attr) {
      return (viewer.getAttribute(attr) || '')
        .split(',')
        .map(function (s) { return s.trim(); })
        .filter(Boolean)
        .map(function (s) {
          var i = s.indexOf(':');
          return i === -1
            ? { key: s, label: s }
            : { key: s.slice(0, i).trim(), label: s.slice(i + 1).trim() };
        });
    }

    // Build one panel (name above image, layer below) into a container.
    function addPanel(container, label) {
      var fig = document.createElement('figure');
      fig.className = 'attn-panel';
      var name = document.createElement('div');
      name.className = 'attn-model-name';
      name.textContent = label;
      var img = document.createElement('img');
      img.className = 'attn-img';
      img.alt = label + ' attention map';
      var cap = document.createElement('figcaption');
      var lyr = document.createElement('span');
      lyr.className = 'attn-layer';
      cap.appendChild(lyr);
      fig.appendChild(name);
      fig.appendChild(img);
      fig.appendChild(cap);
      container.appendChild(fig);
      return { img: img, lyr: lyr };
    }

    var numBlocks = parseInt(viewer.getAttribute('data-num-blocks'), 10) || 23;
    var firstLayer = parseInt(viewer.getAttribute('data-first-layer'), 10) || 1;
    function fallbackLayers() {
      var arr = [];
      for (var k = 0; k < numBlocks; k++) arr.push(firstLayer + k);
      return arr;
    }

    // One or two model row-groups: .attn-models (data-models) and optionally
    // .attn-models2 (data-models2). Both are driven by the same slider.
    var groups = [];
    [['data-models', '.attn-models'], ['data-models2', '.attn-models2']]
      .forEach(function (pair) {
        var container = viewer.querySelector(pair[1]);
        var defs = parseModels(pair[0]);
        if (!container || !defs.length) return;
        defs.forEach(function (m) {
          var p = addPanel(container, m.label);
          m.img = p.img; m.lyr = p.lyr;
          m.kept = numBlocks;
          m.layers = fallbackLayers();   // overwritten by manifest if present
        });
        groups.push.apply(groups, defs);
      });
    if (!groups.length) return;

    function applyManifest(man) {
      groups.forEach(function (m) {
        var info = man && man.models && man.models[m.key];
        if (info) { m.kept = info.kept; m.layers = info.layers; }
      });
      var maxKept = groups.reduce(function (a, m) { return Math.max(a, m.kept); }, 1);
      // Start at the LAST (deepest) layer, where outliers are strongest.
      slider.min = 1; slider.max = maxKept; slider.value = maxKept;
      update();
    }

    function update() {
      var i = parseInt(slider.value, 10);          // 1-based step
      groups.forEach(function (m) {
        var idx = Math.min(i, m.kept);             // clamp to this model's count
        m.img.src = 'static/images/attention/' + m.key + '_block' + pad(idx) + '.png';
        m.lyr.textContent = (m.layers && m.layers[idx - 1] != null)
          ? 'layer ' + m.layers[idx - 1] : '';
      });
      if (output) output.textContent = i;
    }

    slider.addEventListener('input', update);

    // Try to load the manifest; fall back to data-num-blocks if it isn't there.
    fetch('static/images/attention/manifest.json')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(applyManifest)
      .catch(function () { applyManifest(null); });
  });
});

/* ------------- DIFFUSION ATTENTION VIEWER (2D: layer x time) -------------
   Files: static/images/attention_diff/<key>_L<li>_T<ti>.png
   Two sliders (layer, timestep) drive a row of model panels. Per-model layer
   indices + timestep values come from manifest_diff.json (fallback otherwise).
------------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.attn-diff-viewer').forEach(function (viewer) {
    var sL = viewer.querySelector('.attn-slider-layer');
    var sT = viewer.querySelector('.attn-slider-time');
    var oL = viewer.querySelector('.attn-value-layer');
    var oT = viewer.querySelector('.attn-value-time');
    var panels = viewer.querySelector('.attn-diff-models');
    if (!sL || !sT || !panels) return;

    function pad(b) { return (b < 10 ? '0' : '') + b; }

    var models = (viewer.getAttribute('data-models') || '')
      .split(',').map(function (s) { return s.trim(); }).filter(Boolean)
      .map(function (s) {
        var i = s.indexOf(':');
        return i === -1 ? { key: s, label: s }
                        : { key: s.slice(0, i).trim(), label: s.slice(i + 1).trim() };
      });

    models.forEach(function (m) {
      var fig = document.createElement('figure');
      fig.className = 'attn-panel';
      var name = document.createElement('div');
      name.className = 'attn-model-name';
      name.textContent = m.label;
      var img = document.createElement('img');
      img.className = 'attn-img';
      img.alt = m.label + ' attention map';
      var cap = document.createElement('figcaption');
      var lyr = document.createElement('span');
      lyr.className = 'attn-layer';
      cap.appendChild(lyr);
      fig.appendChild(name); fig.appendChild(img); fig.appendChild(cap);
      panels.appendChild(fig);
      m.img = img; m.lyr = lyr;
      m.layers = null;  // per-model real layer ids (from manifest)
    });

    var timesteps = null;     // e.g. [0.1,0.3,0.5,0.7,0.9]

    function update() {
      var li = parseInt(sL.value, 10), ti = parseInt(sT.value, 10);
      models.forEach(function (m) {
        m.img.src = 'static/images/attention_diff/' + m.key +
                    '_L' + pad(li) + '_T' + pad(ti) + '.png';
        m.lyr.textContent = (m.layers && m.layers[li - 1] != null)
          ? 'layer ' + m.layers[li - 1] : '';
      });
      if (oL) oL.textContent = li;
      if (oT) {
        if (timesteps) {
          var tv = timesteps[ti - 1];
          // flow convention: t=1 is the clean image, so lower t = more noise.
          var hint = tv <= 0.34 ? ' (more noise)' : (tv >= 0.66 ? ' (less noise)' : '');
          oT.textContent = 't=' + tv + hint;
        } else {
          oT.textContent = ti;
        }
      }
    }

    function applyManifest(man) {
      if (man) {
        timesteps = man.timesteps || null;
        if (timesteps) { sT.max = timesteps.length; }
        models.forEach(function (m) {
          var info = man.models && man.models[m.key];
          if (info) { m.layers = info.layers; sL.max = info.nlayers; }
        });
      }
      update();
    }

    sL.addEventListener('input', update);
    sT.addEventListener('input', update);
    fetch('static/images/attention_diff/manifest_diff.json')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(applyManifest)
      .catch(function () { applyManifest(null); });
  });
});

/* ------------------- PCA FEATURE-MAP VIEWER (block x time) --------------
   3 panels (input / without registers / with registers); block + noise
   sliders. Files: static/images/pca/{noreg,reg}_b<NN>_t<TT>.png
------------------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.pca-viewer').forEach(function (viewer) {
    var sI = viewer.querySelector('.pca-slider-img');
    var sB = viewer.querySelector('.pca-slider-block');
    var sT = viewer.querySelector('.pca-slider-time');
    var oI = viewer.querySelector('.pca-val-img');
    var oB = viewer.querySelector('.pca-val-block');
    var oT = viewer.querySelector('.pca-val-time');
    var input = viewer.querySelector('#pca-input');
    var noreg = viewer.querySelector('#pca-noreg');
    var reg = viewer.querySelector('#pca-reg');
    if (!sB || !sT || !noreg || !reg) return;

    function pad(n) { return (n < 10 ? '0' : '') + n; }
    var images = ['cat','bellpepper','vase'];
    var blocks = [5, 6, 7];
    var times = null;

    function update() {
      var name = images[(parseInt(sI ? sI.value : 1, 10) - 1)] || images[0];
      var b = parseInt(sB.value, 10), t = parseInt(sT.value, 10);
      var suffix = '_b' + pad(b) + '_t' + pad(t) + '.png';
      if (input) input.src = 'static/images/pca/input_' + name + '.png';
      noreg.src = 'static/images/pca/' + name + '_noreg' + suffix;
      reg.src = 'static/images/pca/' + name + '_reg' + suffix;
      if (oI) oI.textContent = name;
      if (oB) oB.textContent = b;
      if (oT) {
        var tv = times ? times[t - 1] : null;
        oT.textContent = (tv != null) ? ('t=' + tv) : t;
      }
    }

    function applyManifest(man) {
      if (man) {
        if (man.images && man.images.length) { images = man.images; if (sI) sI.max = images.length; }
        if (man.blocks && man.blocks.length) { blocks = man.blocks; sB.min = blocks[0]; sB.max = blocks[blocks.length-1]; }
        if (man.timesteps) { times = man.timesteps; sT.max = times.length; }
      }
      update();
    }

    if (sI) sI.addEventListener('input', update);
    sB.addEventListener('input', update);
    sT.addEventListener('input', update);
    fetch('static/images/pca/manifest_pca.json')
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(applyManifest)
      .catch(function () { applyManifest(null); });
  });
});

/* ---------------- REGISTER GUIDANCE SWITCHER ---------------------
   Three sliders (subject / CFG / RG) pick one generated sample.
   Files: static/images/reg_guidance/<slug>_cfg<c>_rg<r>.png
   Axes come from data-images / data-cfgs / data-rgs on .rg-viewer. */
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('.rg-viewer').forEach(function (viewer) {
    var img = viewer.querySelector('.rg-img');
    var sImg = viewer.querySelector('.rg-slider-img');
    var sCfg = viewer.querySelector('.rg-slider-cfg');
    var sRg = viewer.querySelector('.rg-slider-rg');
    var oImg = viewer.querySelector('.rg-val-img');
    var oCfg = viewer.querySelector('.rg-val-cfg');
    var oRg = viewer.querySelector('.rg-val-rg');
    if (!img || !sImg || !sCfg || !sRg) return;

    var subjects = (viewer.getAttribute('data-images') || '')
      .split(',').map(function (s) { return s.trim(); }).filter(Boolean)
      .map(function (s) {
        var kv = s.split(':');
        return { slug: kv[0].trim(), label: (kv[1] || kv[0]).trim() };
      });
    var cfgs = (viewer.getAttribute('data-cfgs') || '').split(',').map(function (s) { return s.trim(); }).filter(Boolean);
    var rgs = (viewer.getAttribute('data-rgs') || '').split(',').map(function (s) { return s.trim(); }).filter(Boolean);

    sImg.max = subjects.length;
    sCfg.max = cfgs.length;
    sRg.max = rgs.length;

    function update() {
      var subj = subjects[parseInt(sImg.value, 10) - 1] || subjects[0];
      var cfg = cfgs[parseInt(sCfg.value, 10) - 1] || cfgs[0];
      var rg = rgs[parseInt(sRg.value, 10) - 1] || rgs[0];
      img.src = 'static/images/reg_guidance/' + subj.slug + '_cfg' + cfg + '_rg' + rg + '.png';
      if (oImg) oImg.textContent = subj.label;
      if (oCfg) oCfg.textContent = cfg;
      if (oRg) oRg.textContent = rg;
    }

    sImg.addEventListener('input', update);
    sCfg.addEventListener('input', update);
    sRg.addEventListener('input', update);
    update();
  });
});

/* --------------------------- THEME TOGGLE ------------------------- */
(function () {
  var root = document.documentElement;
  var stored = localStorage.getItem('theme');
  var prefersDark = window.matchMedia &&
    window.matchMedia('(prefers-color-scheme: dark)').matches;
  root.setAttribute('data-theme', stored || (prefersDark ? 'dark' : 'light'));

  document.addEventListener('DOMContentLoaded', function () {
    var toggle = document.getElementById('theme-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
    });
  });
})();

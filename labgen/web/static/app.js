/* AI Lab Trainer — клиентская логика (без сборки и внешних зависимостей, кроме локального CodeMirror) */
(function () {
  "use strict";

  var csrf = (document.querySelector('meta[name="csrf-token"]') || {}).content || "";

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function esc(text) {
    return String(text == null ? "" : text).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function postJSON(url, body) {
    return fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf },
      body: JSON.stringify(body || {}),
      credentials: "same-origin",
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        if (!r.ok) throw new Error(data.detail || ("Ошибка " + r.status));
        return data;
      });
    });
  }
  function storage(key, value) {
    try {
      if (value === undefined) return window.localStorage.getItem(key);
      if (value === null) window.localStorage.removeItem(key); else window.localStorage.setItem(key, value);
    } catch (e) { return null; }
  }

  /* --- тема ------------------------------------------------------------------------ */
  $$("[data-theme-toggle]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      document.documentElement.dataset.theme = next;
      storage("labgen:theme", next);
      $$(".CodeMirror").forEach(function (el) { if (el.CodeMirror) el.CodeMirror.refresh(); });
    });
  });

  /* --- выпадающие меню --------------------------------------------------------------- */
  $$("[data-dropdown]").forEach(function (dd) {
    var toggle = $("[data-dropdown-toggle]", dd);
    toggle.addEventListener("click", function (e) { e.stopPropagation(); dd.classList.toggle("open"); });
  });
  document.addEventListener("click", function () { $$("[data-dropdown].open").forEach(function (d) { d.classList.remove("open"); }); });

  /* --- уведомления ------------------------------------------------------------------- */
  $$(".flash").forEach(function (el) {
    setTimeout(function () { el.style.transition = "opacity .3s"; el.style.opacity = "0"; setTimeout(function () { el.remove(); }, 320); }, 5000);
  });

  /* --- кнопки долгих операций ----------------------------------------------------------- */
  $$("form[data-busy]").forEach(function (form) {
    form.addEventListener("submit", function () {
      var btn = $("button[type=submit]", form);
      if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> ' + esc(btn.dataset.busyText || btn.textContent); }
    });
  });
  $$("form[data-confirm]").forEach(function (form) {
    form.addEventListener("submit", function (e) { if (!window.confirm(form.dataset.confirm)) e.preventDefault(); });
  });

  /* --- вкладки ---------------------------------------------------------------------- */
  $$("[data-tabs]").forEach(function (group) {
    var name = group.dataset.tabs;
    var buttons = $$('[data-tab]', group);
    var panes = $$('[data-pane-of="' + name + '"]');
    function show(id, push) {
      buttons.forEach(function (b) { b.classList.toggle("active", b.dataset.tab === id); });
      panes.forEach(function (p) { p.classList.toggle("active", p.dataset.pane === id); });
      if (push && group.hasAttribute("data-hash")) history.replaceState(null, "", "#" + id);
      $$(".CodeMirror", panes.filter(function (p) { return p.dataset.pane === id; })[0] || document.body).forEach(function (el) { el.CodeMirror && el.CodeMirror.refresh(); });
    }
    buttons.forEach(function (b) { b.addEventListener("click", function () { show(b.dataset.tab, true); }); });
    var hash = location.hash.replace("#", "");
    if (group.hasAttribute("data-hash") && hash && buttons.some(function (b) { return b.dataset.tab === hash; })) show(hash, false);
  });

  /* --- редакторы кода в формах ------------------------------------------------------------ */
  function makeEditor(textarea, mode, readOnly) {
    if (!window.CodeMirror) return null;
    var cm = window.CodeMirror.fromTextArea(textarea, {
      mode: mode, theme: "lc", lineNumbers: mode === "python", indentUnit: 4, tabSize: 4, indentWithTabs: false,
      lineWrapping: mode !== "python", matchBrackets: true, styleActiveLine: true, readOnly: readOnly || false,
      extraKeys: { Tab: function (c) { c.replaceSelection("    "); } },
    });
    return cm;
  }
  $$("textarea[data-cm]").forEach(function (ta) {
    var cm = makeEditor(ta, ta.dataset.cm, ta.hasAttribute("readonly"));
    if (cm && ta.dataset.cmHeight) cm.setSize(null, ta.dataset.cmHeight);
  });

  /* --- фильтры списка задач -------------------------------------------------------------- */
  var table = $("table.problems[data-filterable]");
  if (table) {
    var search = $("#filter-search"), diff = $("#filter-diff"), status = $("#filter-status");
    var topic = "";
    function apply() {
      var q = (search.value || "").trim().toLowerCase(), shown = 0;
      $$("tbody tr[data-title]", table).forEach(function (row) {
        var ok = (!q || row.dataset.title.indexOf(q) !== -1) &&
          (!diff.value || row.dataset.level === diff.value) &&
          (!status.value || row.dataset.status === status.value) &&
          (!topic || row.dataset.topic === topic);
        row.classList.toggle("hidden", !ok);
        if (ok) shown++;
      });
      var empty = $("#filter-empty"); if (empty) empty.classList.toggle("hidden", shown > 0);
    }
    [search, diff, status].forEach(function (el) { el.addEventListener("input", apply); });
    $$(".topic-tags .tag").forEach(function (tag) {
      tag.addEventListener("click", function () {
        topic = tag.classList.contains("active") ? "" : tag.dataset.topic;
        $$(".topic-tags .tag").forEach(function (t) { t.classList.toggle("active", t.dataset.topic === topic && topic !== ""); });
        apply();
      });
    });
  }

  /* --- проверка подключения LLM --------------------------------------------------------------- */
  $$("[data-llm-test]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var out = $(btn.dataset.target);
      var label = btn.innerHTML;
      btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Проверка…';
      out.textContent = "";
      postJSON(btn.dataset.llmTest).then(function (data) {
        out.className = "check-result " + (data.ok ? "ok-text" : "err-text");
        out.textContent = (data.ok ? "✓ " : "✗ ") + data.message;
      }).catch(function (e) {
        out.className = "check-result err-text"; out.textContent = "✗ " + e.message;
      }).finally(function () { btn.disabled = false; btn.innerHTML = label; });
    });
  });

  /* --- прогресс генерации ------------------------------------------------------------------- */
  var jobRoot = $("[data-job]");
  if (jobRoot) {
    var jobId = jobRoot.dataset.job;
    var order = JSON.parse(jobRoot.dataset.nodes);
    var labels = JSON.parse(jobRoot.dataset.labels);
    function render(job) {
      var seen = {}, lastNode = null;
      job.steps.forEach(function (s) { seen[s.node] = (seen[s.node] || []).concat(s.log); lastNode = s.node; });
      var html = "";
      var nodes = order.filter(function (n) { return seen[n] || (n !== "catalog" && n !== "generate" && n !== "finalize"); });
      if (job.steps.some(function (s) { return s.node === "catalog"; })) nodes = nodes.filter(function (n) { return n !== "generate"; });
      nodes.forEach(function (node, i) {
        var done = !!seen[node];
        var cls = done ? "done" : "";
        if (!done && job.status === "running" && (i === 0 || seen[nodes[i - 1]])) cls = "current";
        if (job.status === "error" && !done && (i === 0 || seen[nodes[i - 1]])) cls = "error";
        var log = (seen[node] || []).slice(-4).map(function (l) { return '<div class="log">' + esc(l) + "</div>"; }).join("");
        html += '<li class="' + cls + '"><span class="marker">' + (cls === "done" ? "✓" : cls === "error" ? "!" : i + 1) +
          "</span><div><div>" + esc(labels[node] || node) + (cls === "current" ? ' <span class="spinner" style="width:11px;height:11px;margin-left:4px"></span>' : "") +
          "</div>" + log + "</div></li>";
      });
      $("#job-steps").innerHTML = html;
      $("#job-elapsed").textContent = job.elapsed.toFixed(1) + " с";
      if (job.status === "done") {
        $("#job-status").innerHTML = '<span class="pill ok">Готово</span>';
        setTimeout(function () { location.href = "/labs/" + job.lab_id; }, 700);
      } else if (job.status === "error") {
        $("#job-status").innerHTML = '<span class="pill err">Ошибка</span>';
        var err = $("#job-error"); err.classList.remove("hidden"); err.textContent = job.error;
        $("#job-actions").classList.remove("hidden");
      }
      return job.status === "running";
    }
    (function poll() {
      fetch("/api/jobs/" + jobId, { credentials: "same-origin" }).then(function (r) { return r.json(); }).then(function (job) {
        if (render(job)) setTimeout(poll, 700);
      }).catch(function () { setTimeout(poll, 1500); });
    })();
  }

  /* --- рабочее пространство задачи ------------------------------------------------------------- */
  var dataEl = $("#lab-data");
  if (dataEl) {
    var lab = JSON.parse(dataEl.textContent);
    var storeKey = "labgen:code:" + lab.id + ":v" + lab.version;
    var files = {
      solution: { text: storage(storeKey) || lab.starter, readOnly: false },
      reference: { text: lab.reference, readOnly: true },
      public: { text: lab.tests_public, readOnly: true },
      hidden: { text: lab.tests_hidden, readOnly: true },
    };
    var current = "solution";
    var area = $("#editor-area");
    var cm = makeEditor(area, "python", false);
    var docs = {};
    if (cm) {
      Object.keys(files).forEach(function (k) { docs[k] = window.CodeMirror.Doc(files[k].text, "python"); });
      cm.swapDoc(docs.solution);
      cm.on("change", function () { if (current === "solution") storage(storeKey, cm.getValue()); });
    } else {
      area.value = files.solution.text;
      area.addEventListener("input", function () { if (current === "solution") { files.solution.text = area.value; storage(storeKey, area.value); } });
    }
    function code() { return cm ? docs.solution.getValue() : files.solution.text; }
    $$("[data-file]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        current = btn.dataset.file;
        $$("[data-file]").forEach(function (b) { b.classList.toggle("active", b === btn); });
        if (cm) { cm.swapDoc(docs[current]); cm.setOption("readOnly", files[current].readOnly); }
        else { area.value = current === "solution" ? files.solution.text : files[current].text; area.readOnly = files[current].readOnly; }
        $("#readonly-badge").classList.toggle("hidden", !files[current].readOnly);
      });
    });
    var reset = $("#reset-code");
    if (reset) reset.addEventListener("click", function () {
      if (!window.confirm("Вернуть шаблон решения? Текущий код будет потерян.")) return;
      storage(storeKey, null);
      if (cm) docs.solution.setValue(lab.starter); else { files.solution.text = lab.starter; area.value = lab.starter; }
    });

    var consoleBody = $("#console-result");
    function showConsole(tab) {
      $$('[data-pane-of="console"]').forEach(function (p) { p.classList.toggle("active", p.dataset.pane === tab); });
      $$('[data-tabs="console"] [data-tab]').forEach(function (b) { b.classList.toggle("active", b.dataset.tab === tab); });
    }
    function busy(btn, on, text) {
      if (on) { btn.dataset.label = btn.innerHTML; btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> ' + text; }
      else { btn.disabled = false; btn.innerHTML = btn.dataset.label; }
    }
    function renderCases(data) {
      var all = data.passed === data.total && data.total > 0;
      var html = '<h3 class="' + (all ? "ok-text" : "err-text") + '">' + (all ? "Публичные тесты пройдены" : "Есть ошибки") +
        '</h3><div class="muted">Пройдено ' + data.passed + " из " + data.total + ' публичных тестов</div><div class="case-tabs">';
      data.cases.forEach(function (c, i) {
        html += '<button class="case-tab' + (i === 0 ? " active" : "") + '" data-case="' + i + '"><span class="dot ' + (c.passed ? "on" : "err") + '"></span>' + esc(c.name.replace(/^test_/, "")) + "</button>";
      });
      html += '</div><div class="case-box" id="case-box"></div>';
      consoleBody.innerHTML = html;
      function pick(i) {
        var c = data.cases[i];
        $$(".case-tab", consoleBody).forEach(function (b) { b.classList.toggle("active", b.dataset.case === String(i)); });
        $("#case-box").textContent = c.passed ? "✓ " + c.name + " — пройден" : "✗ " + c.name + "\n" + (c.message || "тест не пройден");
      }
      $$(".case-tab", consoleBody).forEach(function (b) { b.addEventListener("click", function () { pick(+b.dataset.case); }); });
      var firstFail = data.cases.findIndex(function (c) { return !c.passed; });
      if (data.cases.length) pick(firstFail >= 0 ? firstFail : 0);
    }
    function renderGrade(r) {
      var full = r.auto_score === r.auto_max, none = r.auto_score === 0;
      var title = r.infrastructure_error ? "Ошибка выполнения" : full ? "Принято" : none ? "Не принято" : "Частично принято";
      var cls = full ? "ok-text" : none || r.infrastructure_error ? "err-text" : "warn-text";
      var html = '<h3 class="' + cls + '">' + title + '</h3><div class="muted">' + r.auto_score + " / " + r.auto_max +
        " баллов автопроверки · тестов " + r.tests_passed + "/" + r.tests_total +
        ' · <a href="' + r.url + '">подробный отчёт</a></div>';
      if (r.infrastructure_error) html += '<div class="case-box" style="margin-top:10px">' + esc(r.infrastructure_error) + "</div>";
      html += '<div style="margin-top:10px">';
      r.requirements.forEach(function (g) {
        var icon = !g.auto ? '<span class="muted">◇</span>' : g.tests_passed === g.tests_total ? '<span class="ok-text">✓</span>' : '<span class="err-text">✗</span>';
        var msg = g.failures && g.failures.length ? '<div class="muted small mono">' + esc(g.failures[0]) + "</div>" : "";
        html += '<div class="result-req">' + icon + "<div><strong>" + g.id + "</strong> " + (g.auto ? "" : '<span class="muted">проверяет преподаватель</span>') + msg +
          '</div><span class="score">' + (g.auto ? g.earned + " / " + g.points : "— / " + g.points) + "</span></div>";
      });
      consoleBody.innerHTML = html + "</div>";
    }
    var runBtn = $("#run-btn"), submitBtn = $("#submit-btn");
    runBtn.addEventListener("click", function () {
      busy(runBtn, true, "Запуск…"); showConsole("result");
      consoleBody.innerHTML = '<div class="muted"><span class="spinner"></span> Публичные тесты выполняются в песочнице…</div>';
      postJSON("/api/labs/" + lab.id + "/run", { code: code() }).then(renderCases).catch(function (e) {
        consoleBody.innerHTML = '<div class="err-text">' + esc(e.message) + "</div>";
      }).finally(function () { busy(runBtn, false); });
    });
    submitBtn.addEventListener("click", function () {
      busy(submitBtn, true, "Проверка…"); showConsole("result");
      consoleBody.innerHTML = '<div class="muted"><span class="spinner"></span> Полная автопроверка: публичные и скрытые тесты…</div>';
      postJSON("/api/labs/" + lab.id + "/submit", { code: code() }).then(renderGrade).catch(function (e) {
        consoleBody.innerHTML = '<div class="err-text">' + esc(e.message) + "</div>";
      }).finally(function () { busy(submitBtn, false); });
    });
  }
})();

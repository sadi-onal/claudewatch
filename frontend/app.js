function appRoot() {
  return {
    view: "dashboard",
    sessions: [],
    history: [],
    stats: {},
    health: { iterm_api: true, tmux_available: true, log_dir_found: true, issues: [] },
    config: { pricing: {} },
    filter: "All",
    filters: ["All", "iTerm", "Tmux", "Headless", "Working", "Idle", "High-cost"],
    detailPid: null,
    detail: null,
    copiedId: null,
    showNewModal: false,
    newSession: { cwd: "", window_type: "new-window", skipPerm: true, customFlags: "" },
    newSessionError: "",
    _sse: null,

    async init() {
      await Promise.all([this.loadHealth(), this.loadSessions(), this.loadStats(), this.loadConfig()]);
      this.connectSSE();
      this._startNowTimer();
      setInterval(() => this.loadStats(), 5000);
      setInterval(() => this.loadHealth(), 30000);
    },

    async loadHealth() {
      try {
        const r = await fetch("/api/health");
        if (r.ok) this.health = await r.json();
      } catch (e) { console.warn("health failed", e); }
    },
    async loadSessions() {
      try {
        const r = await fetch("/api/sessions");
        if (r.ok) this.sessions = await r.json();
      } catch (e) { console.warn("sessions failed", e); }
    },
    async loadStats() {
      try {
        const r = await fetch("/api/stats");
        if (r.ok) this.stats = await r.json();
      } catch (e) { /* ignore */ }
    },
    async loadHistory() {
      try {
        const r = await fetch("/api/history");
        if (r.ok) this.history = await r.json();
      } catch (e) { /* ignore */ }
    },
    async loadConfig() {
      try {
        const r = await fetch("/api/config");
        if (r.ok) this.config = await r.json();
      } catch (e) { /* ignore */ }
    },
    async saveConfig(updates) {
      try {
        const r = await fetch("/api/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(updates),
        });
        if (r.ok) this.config = await r.json();
      } catch (e) { console.warn("save config failed", e); }
    },
    async updatePricing(model, key, value) {
      const v = parseFloat(value);
      if (Number.isNaN(v)) return;
      const next = { ...(this.config.pricing || {}) };
      next[model] = { ...(next[model] || {}), [key]: v };
      this.config.pricing = next;
      await this.saveConfig({ pricing: next });
    },
    async loadDetail(pid) {
      try {
        const r = await fetch(`/api/sessions/${pid}`);
        if (r.ok) this.detail = await r.json();
      } catch (e) { /* ignore */ }
    },

    async copyId(id) {
      if (!id) return;
      try {
        await navigator.clipboard.writeText(id);
        this.copiedId = id;
        setTimeout(() => { if (this.copiedId === id) this.copiedId = null; }, 1500);
      } catch (e) { /* clipboard unavailable */ }
    },

    connectSSE() {
      try {
        this._sse = new EventSource("/api/stream");
        this._sse.addEventListener("snapshot", (e) => {
          const data = JSON.parse(e.data);
          if (data.sessions) this.sessions = data.sessions;
        });
        this._sse.addEventListener("session.started", (e) => {
          const d = JSON.parse(e.data);
          if (d.session) this._upsertSession(d.session);
        });
        this._sse.addEventListener("session.updated", (e) => {
          const d = JSON.parse(e.data);
          if (d.session) this._upsertSession(d.session);
        });
        this._sse.addEventListener("session.ended", (e) => {
          const d = JSON.parse(e.data);
          if (d.pid) this.sessions = this.sessions.filter((s) => s.pid !== d.pid);
        });
        this._sse.onerror = () => {
          if (this._sse) this._sse.close();
          setTimeout(() => this.connectSSE(), 3000);
        };
      } catch (e) { console.warn("SSE failed", e); }
    },
    _upsertSession(sess) {
      const idx = this.sessions.findIndex((s) => s.pid === sess.pid);
      if (idx >= 0) this.sessions.splice(idx, 1, sess);
      else this.sessions.push(sess);
    },

    visibleSessions() {
      let arr = [...this.sessions];
      switch (this.filter) {
        case "iTerm": arr = arr.filter((s) => s.location_type === "iterm"); break;
        case "Tmux": arr = arr.filter((s) => s.location_type === "tmux"); break;
        case "Headless": arr = arr.filter((s) => s.location_type === "headless"); break;
        case "Working": arr = arr.filter((s) => s.status === "working" || s.is_in_flight); break;
        case "Idle": arr = arr.filter((s) => s.status === "idle" && !s.is_in_flight); break;
        case "High-cost":
          arr = arr.filter((s) => s.usage && (s.usage.cost_estimate_usd || 0) >= 1.0);
          break;
      }
      // Most recently-active first
      arr.sort((a, b) => (b.last_activity_at || "").localeCompare(a.last_activity_at || ""));
      return arr;
    },

    nowTick: 0,
    _startNowTimer() {
      if (this._tickTimer) return;
      this._tickTimer = setInterval(() => { this.nowTick = Date.now(); }, 1000);
    },

    fmtElapsed(seconds, fromIso) {
      // Live-updating elapsed seconds: use base seconds + delta since fromIso was captured.
      void this.nowTick;  // make Alpine track this
      let s = Number(seconds || 0);
      if (fromIso) {
        const driftMs = Date.now() - new Date(fromIso).getTime();
        if (Number.isFinite(driftMs) && driftMs > 0) s = Math.floor(driftMs / 1000);
      }
      if (s < 0) s = 0;
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      const sec = s % 60;
      if (h > 0) return `${h}h ${m}m ${sec}s`;
      if (m > 0) return `${m}m ${sec}s`;
      return `${sec}s`;
    },

    relTime(iso) {
      void this.nowTick;
      if (!iso) return "—";
      const delta = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
      if (delta < 5) return "just now";
      if (delta < 60) return `${delta}s ago`;
      if (delta < 3600) return `${Math.floor(delta / 60)}m ${delta % 60}s ago`;
      const h = Math.floor(delta / 3600);
      return `${h}h ${Math.floor((delta % 3600) / 60)}m ago`;
    },

    fmtNum(n) {
      if (n === null || n === undefined) return "—";
      if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
      if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
      if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
      return String(n);
    },
    fmtMoney(n) {
      if (n === null || n === undefined) return "—";
      return "$" + Number(n).toFixed(2);
    },
    fmtDuration(s) {
      if (!s) return "0s";
      const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
      if (h) return `${h}h ${m}m`;
      if (m) return `${m}m`;
      return `${s}s`;
    },
    fmtTime(iso) {
      if (!iso) return "—";
      return new Date(iso).toLocaleString();
    },

    async focusSession(sess) {
      try {
        const r = await fetch(`/api/sessions/${sess.pid}/focus`, { method: "POST" });
        if (!r.ok) alert("Focus failed: " + r.status);
      } catch (e) { alert("Focus error: " + e); }
    },
    async haltSession(sess) {
      if (!confirm(`Halt PID ${sess.pid} in ${sess.cwd}?\nUnsaved work may be lost.`)) return;
      try {
        const r = await fetch(`/api/sessions/${sess.pid}/halt`, { method: "POST" });
        if (!r.ok) alert("Halt failed: " + r.status);
      } catch (e) { alert("Halt error: " + e); }
    },
    openNewFromCwd(cwd) {
      this.newSession.cwd = cwd || "";
      this.showNewModal = true;
      this.newSessionError = "";
    },
    async submitNew() {
      this.newSessionError = "";
      const flags = [];
      if (this.newSession.skipPerm) flags.push("--dangerously-skip-permissions");
      if (this.newSession.customFlags) {
        for (const tok of this.newSession.customFlags.split(/\s+/).filter(Boolean)) {
          flags.push(tok);
        }
      }
      try {
        const r = await fetch("/api/sessions/new", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            cwd: this.newSession.cwd,
            window_type: this.newSession.window_type,
            flags,
            command: "claude",
          }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({}));
          this.newSessionError = err.detail || `HTTP ${r.status}`;
          return;
        }
        this.showNewModal = false;
      } catch (e) {
        this.newSessionError = String(e);
      }
    },
  };
}

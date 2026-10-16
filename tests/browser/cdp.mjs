// Minimal Chrome DevTools Protocol client using only Node built-ins (global WebSocket, Node 22+).
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

export function findChrome() {
  const candidates = [
    process.env.PCUR_CHROME,
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser",
  ].filter(Boolean);
  return candidates.find((p) => existsSync(p)) || null;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class Browser {
  static async launch({ width = 1400, height = 900 } = {}) {
    const chrome = findChrome();
    if (!chrome) throw new Error("no Chrome or Chromium found (set PCUR_CHROME)");
    const dir = mkdtempSync(join(tmpdir(), "pcur-chrome-"));
    const proc = spawn(chrome, [
      "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run", "--no-default-browser-check",
      "--remote-debugging-port=0", `--user-data-dir=${dir}`, `--window-size=${width},${height}`, "about:blank",
    ], { stdio: ["ignore", "ignore", "pipe"] });
    const wsUrl = await new Promise((resolve, reject) => {
      let buf = "";
      const t = setTimeout(() => reject(new Error("Chrome did not start: " + buf.slice(-300))), 20000);
      proc.stderr.on("data", (d) => {
        buf += d.toString();
        const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
        if (m) { clearTimeout(t); resolve(m[1]); }
      });
      proc.on("exit", (c) => reject(new Error("Chrome exited early (" + c + ")")));
    });
    const b = new Browser(proc, dir, wsUrl);
    await b.connect();
    return b;
  }

  constructor(proc, dir, wsUrl) {
    Object.assign(this, { proc, dir, wsUrl, nextId: 1, pending: new Map(), listeners: [] });
  }

  async connect() {
    this.ws = new WebSocket(this.wsUrl);
    await new Promise((res, rej) => { this.ws.onopen = res; this.ws.onerror = () => rej(new Error("websocket failed")); });
    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
      } else if (msg.method) {
        this.listeners.forEach((l) => l(msg));
      }
    };
  }

  send(method, params = {}, sessionId) {
    const id = this.nextId++;
    this.ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }

  async newPage({ isolated = false } = {}) {
    let browserContextId;
    if (isolated) ({ browserContextId } = await this.send("Target.createBrowserContext"));
    const { targetId } = await this.send("Target.createTarget", { url: "about:blank", ...(browserContextId ? { browserContextId } : {}) });
    const { sessionId } = await this.send("Target.attachToTarget", { targetId, flatten: true });
    const page = new Page(this, sessionId, targetId);
    await page.send("Page.enable");
    await page.send("Runtime.enable");
    return page;
  }

  async close() {
    try { this.ws.close(); } catch { /* already closed */ }
    this.proc.kill("SIGKILL");
    await sleep(100);
    try { rmSync(this.dir, { recursive: true, force: true }); } catch { /* best effort */ }
  }
}

export class Page {
  constructor(browser, sessionId, targetId) {
    Object.assign(this, { browser, sessionId, targetId, console: [], errors: [] });
    browser.listeners.push((m) => {
      if (m.sessionId !== sessionId) return;
      if (m.method === "Runtime.consoleAPICalled") this.console.push(m.params.args.map((a) => a.value ?? a.description).join(" "));
      if (m.method === "Runtime.exceptionThrown") this.errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    });
  }

  send(method, params) { return this.browser.send(method, params, this.sessionId); }

  async goto(url) {
    await this.send("Page.navigate", { url });
    await this.waitFor("document.readyState === 'complete'", 20000);
  }

  async eval(expression) {
    const r = await this.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  }

  async waitFor(expression, timeout = 10000, label = expression) {
    const end = Date.now() + timeout;
    let last;
    while (Date.now() < end) {
      try { last = await this.eval(expression); if (last) return last; } catch { /* page still loading */ }
      await sleep(40);
    }
    throw new Error(`timed out waiting for: ${label}`);
  }

  async frames(n = 2) { await this.eval(`new Promise((r) => { let n = ${n}; const f = () => (--n <= 0 ? r(true) : requestAnimationFrame(f)); requestAnimationFrame(f); })`); }

  async key(key, { code, vk, text, mods = 0 } = {}) {
    const base = { key, code: code || key, modifiers: mods, windowsVirtualKeyCode: vk || (key.length === 1 ? key.toUpperCase().charCodeAt(0) : 0),
      ...(text !== undefined ? { text } : key.length === 1 ? { text: key } : {}) };
    await this.send("Input.dispatchKeyEvent", { type: "keyDown", ...base });
    await this.send("Input.dispatchKeyEvent", { type: "keyUp", ...base });
  }

  async viewport(width, height) {
    await this.send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
  }

  async screenshot(path) {
    const { data } = await this.send("Page.captureScreenshot", { format: "png" });
    const { writeFileSync } = await import("node:fs");
    writeFileSync(path, Buffer.from(data, "base64"));
  }
}

export { sleep };

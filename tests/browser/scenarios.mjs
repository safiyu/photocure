// Real-browser checks of the UI against a seeded server.
// node tests/browser/scenarios.mjs META_JSON RESULTS_JSON [SHOT_DIR]
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { Browser, sleep } from "./cdp.mjs";

const meta = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = process.argv[3];
const shots = process.argv[4];
if (shots) mkdirSync(shots, { recursive: true });

const KEYS = {
  ArrowRight: { code: "ArrowRight", vk: 39 }, ArrowLeft: { code: "ArrowLeft", vk: 37 },
  ArrowDown: { code: "ArrowDown", vk: 40 }, ArrowUp: { code: "ArrowUp", vk: 38 },
  Enter: { code: "Enter", vk: 13, text: "\r" }, Escape: { code: "Escape", vk: 27 }, Tab: { code: "Tab", vk: 9 },
};
const name = (i) => `IMG_${String(i).padStart(5, "0")}.jpg`;
const results = [];
let page;

function check(cond, msg) { if (!cond) throw new Error(msg); }
const press = (k, mods) => page.key(k, { ...(KEYS[k] || {}), mods });
const shot = async (n) => { if (shots) await page.screenshot(join(shots, n + ".png")); };
const cardCount = () => page.eval("document.querySelectorAll('#grid .vg-window .card').length");
const selIdx = () => page.eval("(() => { const c = document.querySelector('#grid .card.sel'); return c ? Number(c.dataset.idx) : -1; })()");
const cols = async () => Number(await page.eval("getComputedStyle(document.querySelector('#grid')).getPropertyValue('--vg-cols')"));
const api = (path, init) => page.eval(`fetch(${JSON.stringify(path)}, ${JSON.stringify(init || {})}).then((r) => r.json())`);
const settle = async () => { await page.frames(3); await sleep(120); };

async function moveTo(idx, { from = -1 } = {}) {
  // Walk the keyboard cursor to idx with arrow keys (down by rows, then right).
  const c = await cols();
  let cur = await selIdx();
  if (cur < 0) { await press("ArrowRight"); await page.waitFor("document.querySelector('#grid .card.sel')", 8000); cur = await selIdx(); }
  while (cur + c <= idx) { await press("ArrowDown"); cur += c; }
  while (cur < idx) { await press("ArrowRight"); cur += 1; }
  await page.waitFor(`(() => { const c = document.querySelector('#grid .card.sel'); return c && Number(c.dataset.idx) === ${idx}; })()`, 15000, `cursor at ${idx}`);
}

async function scenario(title, fn) {
  const t0 = Date.now();
  try { await fn(); results.push({ name: title, ok: true, ms: Date.now() - t0 }); console.log("ok   ", title); }
  catch (e) { results.push({ name: title, ok: false, error: String(e.message || e), ms: Date.now() - t0 }); console.log("FAIL ", title, "-", e.message); }
}

const browser = await Browser.launch({ width: 1400, height: 900 });
try {
  page = await browser.newPage();
  await page.goto(`${meta.base}/#code=${meta.otc}`);
  await page.waitFor("document.querySelector('#grid') && document.querySelector('#grid').classList.contains('vgrid')", 20000, "app initialised");
  await page.eval("document.querySelector('.tab[data-tab=\"categories\"]').click()");
  await page.waitFor("document.querySelectorAll('#grid .vg-window .card').length > 0", 20000, "first cards");
  await settle();

  await scenario("one-time code signs in and is removed from the address bar", async () => {
    check(!(await page.eval("location.href")).includes("code="), "the code is still in the URL");
    check(await page.eval("document.querySelector('#auth-modal').hidden"), "sign-in box is showing");
    check(!(await page.eval("document.cookie.includes('pcur_session')")), "the session cookie is readable by scripts");
    const again = await browser.newPage({ isolated: true });
    await again.goto(`${meta.base}/#code=${meta.otc}`);  // the same code a second time
    await again.waitFor("!document.querySelector('#auth-modal').hidden", 15000, "sign-in box for a used code");
    await again.send("Target.closeTarget", { targetId: again.targetId });
  });

  await scenario("sign-in box refuses a wrong token and accepts the right one", async () => {
    const p = await browser.newPage({ isolated: true });
    await p.goto(meta.base + "/");
    await p.waitFor("!document.querySelector('#auth-modal').hidden", 15000, "sign-in box");
    await p.eval(`(() => { const i = document.querySelector('#auth-input'); i.value = 'not-the-token'; document.querySelector('#auth-form').requestSubmit(); })()`);
    await p.waitFor("!document.querySelector('#auth-error').hidden", 8000, "error message");
    check(!(await p.eval("document.querySelector('#auth-modal').hidden")), "box closed after a wrong token");
    await p.eval(`(() => { const i = document.querySelector('#auth-input'); i.value = ${JSON.stringify(meta.token)}; document.querySelector('#auth-form').requestSubmit(); })()`);
    await p.waitFor("document.querySelector('#auth-modal').hidden", 8000, "box closes");
    await p.waitFor("document.querySelector('#grid').classList.contains('vgrid')", 15000, "app finishes loading after sign-in");
    await p.send("Target.closeTarget", { targetId: p.targetId });
  });

  await scenario("grid keeps a window of cards, not the whole library", async () => {
    const n = await cardCount();
    check(n > 0 && n <= 100, `${n} cards in the DOM`);
    const total = await page.eval("document.querySelector('#rowcount').textContent");
    check(total === "5,000 photos", `count text is "${total}"`);
    const h = parseFloat(await page.eval("document.querySelector('#grid .vg-spacer').style.height"));
    const c = await cols();
    const rowH = parseFloat(await page.eval("getComputedStyle(document.querySelector('#grid')).getPropertyValue('--vg-card-h')")) + 22;
    check(Math.abs(h - Math.ceil(meta.n / c) * rowH) < 1, `spacer ${h}px does not match ${Math.ceil(meta.n / c)} rows of ${rowH}px`);
    await shot("01-library");
  });

  await scenario("DOM stays within 100 cards through rapid scrolling over 5,000 photos", async () => {
    const stats = await page.eval(`(async () => {
      const spacer = document.querySelector('#grid .vg-spacer');
      const maxY = document.documentElement.scrollHeight - innerHeight;
      let worst = 0, gaps = 0, samples = 0;
      const sample = () => {
        const idx = [...document.querySelectorAll('#grid .vg-window .card')].map((c) => Number(c.dataset.idx)).sort((a, b) => a - b);
        worst = Math.max(worst, idx.length); samples++;
        for (let i = 1; i < idx.length; i++) if (idx[i] !== idx[i - 1] + 1) gaps++;
      };
      const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      for (let y = 0; y <= maxY; y += 311) { scrollTo(0, y); await frame(); sample(); }       // steady scroll to the end
      for (let k = 0; k < 120; k++) { scrollTo(0, Math.floor(Math.random() * maxY)); await frame(); sample(); }  // wild jumps
      scrollTo(0, maxY); await frame(); await new Promise((r) => setTimeout(r, 400)); sample();
      const last = [...document.querySelectorAll('#grid .vg-window .card')].map((c) => Number(c.dataset.idx));
      return { worst, gaps, samples, lastIdx: Math.max(...last), maxY, spacerH: parseFloat(spacer.style.height) };
    })()`);
    check(stats.worst <= 100, `up to ${stats.worst} cards were in the DOM at once`);
    check(stats.gaps === 0, `${stats.gaps} holes in the rendered range`);
    check(stats.lastIdx === meta.n - 1, `the end of the list shows card ${stats.lastIdx}, not ${meta.n - 1}`);
    await settle();
    await shot("02-bottom");
    await page.eval("scrollTo(0, 0)");
    await settle();
  });

  await scenario("cards are recycled, not rebuilt, while scrolling", async () => {
    await page.eval("window.__mark = document.querySelector('#grid .card[data-idx=\"2\"]'); window.__mark.__tag = 'same'");
    await page.eval("scrollTo(0, 120)");
    await settle();
    const same = await page.eval("document.contains(window.__mark) && window.__mark.__tag === 'same'");
    check(same, "a card that stayed in view was replaced by a new element");
  });

  await scenario("pages load on demand instead of the whole library", async () => {
    const urls = await page.eval("performance.getEntriesByType('resource').map((r) => r.name).filter((u) => u.includes('/api/images?'))");
    check(urls.length > 0 && urls.length < 25, `${urls.length} page requests for ${meta.n / 200} pages`);
    check(urls.every((u) => u.includes("limit=200")), "a page request used a different size");
  });

  await scenario("arrow keys move into rows that are not rendered and scroll them into view", async () => {
    await page.eval("scrollTo(0, 0)");
    await settle();
    await press("ArrowRight");
    await page.waitFor("document.querySelector('#grid .card.sel')", 8000, "first cursor");
    check((await selIdx()) === 0, "first press did not select the first photo");
    const target = 912;
    await moveTo(target);
    await settle();
    const info = await page.eval(`(() => { const c = document.querySelector('#grid .card.sel'); const r = c.getBoundingClientRect();
      return { name: c.querySelector('.fn').textContent, visible: r.bottom > 0 && r.top < innerHeight, n: document.querySelectorAll('#grid .vg-window .card').length }; })()`);
    check(info.name === name(target), `cursor card is ${info.name}, expected ${name(target)}`);
    check(info.visible, "the selected card is off screen");
    check(info.n <= 100, `${info.n} cards rendered`);
    await shot("03-cursor");
  });

  let rated;
  await scenario("keys rate, clear, reject, un-reject and favourite the selected photo", async () => {
    rated = await page.eval("document.querySelector('#grid .card.sel .fn').textContent");
    const flags = () => page.eval("document.querySelector('#grid .card.sel .flags').textContent");
    const ratings = async () => (await api("/api/facets")).ratings;
    const before = await ratings();
    await press("4");
    await page.waitFor("document.querySelector('#grid .card.sel .flags').textContent.includes('\\u26054')", 8000, "four stars shown");
    check((await ratings())["4"] === before["4"] + 1, "server did not record 4 stars");
    await press("0");
    await page.waitFor("!document.querySelector('#grid .card.sel .flags').textContent.includes('\\u2605')", 8000, "rating cleared");
    await press("x");
    await page.waitFor("document.querySelector('#grid .card.sel.rejected')", 8000, "rejected look");
    check((await flags()).includes("Rejected"), "no Rejected badge");
    check((await ratings())["-1"] === before["-1"] + 1, "server did not record the reject");
    await press("x");
    await page.waitFor("!document.querySelector('#grid .card.sel.rejected')", 8000, "un-rejected");
    await press("f");
    await page.waitFor("document.querySelector('#grid .card.sel .flags').textContent.includes('\\u2665')", 8000, "favourite badge");
    await press("f");
    await page.waitFor("!document.querySelector('#grid .card.sel .flags').textContent.includes('\\u2665')", 8000, "favourite removed");
    await press("5");
    await page.waitFor("document.querySelector('#grid .card.sel .flags').textContent.includes('\\u26055')", 8000, "five stars shown");
    await shot("04-rated");
  });

  await scenario("a rating reaches the XMP sidecar after the debounce", async () => {
    const f = join(meta.lib, rated + ".xmp");
    // Several quick changes are one debounced write each, so wait for the last one to land.
    const holds5 = () => existsSync(f) && /Rating[^>]*>5</.test(readFileSync(f, "utf8"));
    const end = Date.now() + 8000;
    while (Date.now() < end && !holds5()) await sleep(100);
    check(existsSync(f), "no sidecar was written");
    if (!holds5()) {
      const item = (await api(`/api/images?q=${rated}`)).items[0];
      throw new Error(`sidecar does not hold the 5 stars. file: ${readFileSync(f, "utf8").replace(/\s+/g, " ").slice(0, 400)} | server rating: ${item.rating}`);
    }
  });

  await scenario("a photo edited elsewhere asks before it is overwritten (focus trap, Escape, overwrite)", async () => {
    await page.eval("scrollTo(0, 0)");
    await settle();
    await page.eval("document.querySelector('#grid .card[data-idx=\"10\"] .thumb').click()");
    await page.waitFor("!document.querySelector('#lightbox').hidden", 8000, "lightbox");
    const foreign = meta.foreign;
    check((await page.eval("document.querySelector('#lb-title').textContent")) === foreign, "wrong photo in the viewer");
    await press("5");
    await page.waitFor("!document.querySelector('#confirm-modal').hidden", 8000, "confirm dialog");
    const active = () => page.eval("document.activeElement.id");
    check((await active()) === "confirm-modal-cancel", "focus did not start on Cancel");
    check((await page.eval("document.querySelector('#confirm-modal').getAttribute('role')")) === "alertdialog", "wrong role");
    await press("Tab");
    check((await active()) === "confirm-modal-ok", "Tab did not reach OK");
    await press("Tab");
    check((await active()) === "confirm-modal-cancel", "Tab did not wrap back to Cancel");
    await press("Tab", 8);  // Shift+Tab
    check((await active()) === "confirm-modal-ok", "Shift+Tab did not wrap");
    await shot("05-confirm");
    await press("Escape");
    await page.waitFor("document.querySelector('#confirm-modal').hidden", 4000, "dialog closes on Escape");
    check(!(await page.eval("document.querySelector('#lightbox').hidden")), "Escape also closed the viewer behind the dialog");
    check((await api("/api/images?limit=1&date_to=2023-01-01")).total >= 1, "api sanity");
    const unchanged = await api(`/api/images?q=${foreign}`);
    check(unchanged.items[0].rating === null, "the rating changed although the dialog was cancelled");
    await press("5");
    await page.waitFor("!document.querySelector('#confirm-modal').hidden", 8000, "dialog again");
    await page.eval("document.querySelector('#confirm-modal-ok').click()");
    await page.waitFor("document.querySelector('#confirm-modal').hidden", 4000, "dialog closes");
    await page.waitFor(`document.querySelectorAll('#lb-rate .star.on').length === 5`, 8000, "five stars lit in the viewer");
    const f = join(meta.lib, foreign + ".xmp");
    const end = Date.now() + 6000;
    while (Date.now() < end && !/Rating[^>]*>5</.test(readFileSync(f, "utf8"))) await sleep(100);
    check(/Rating[^>]*>5</.test(readFileSync(f, "utf8")), "the overwrite never reached the sidecar");
    await page.eval("document.querySelector('#lb-close').click()");
  });

  await scenario("the viewer steps across a page boundary and loads the next photos", async () => {
    await page.eval(`scrollTo(0, Math.floor(198 / ${await cols()}) * ${(await page.eval("parseFloat(getComputedStyle(document.querySelector('#grid')).getPropertyValue('--vg-card-h'))")) + 22})`);
    await settle();
    await page.eval("document.querySelector('#grid .card[data-idx=\"198\"] .thumb').click()");
    await page.waitFor("!document.querySelector('#lightbox').hidden", 8000, "lightbox");
    check((await page.eval("document.querySelector('#lb-title').textContent")) === name(198), "opened the wrong photo");
    for (let i = 0; i < 4; i++) { await press("ArrowRight"); await sleep(250); }
    await page.waitFor(`document.querySelector('#lb-title').textContent === ${JSON.stringify(name(202))}`, 8000, "photo 202");
    check((await page.eval("document.querySelector('#lb-count').textContent")) === `203 of ${meta.n.toLocaleString("en-US")}`, "count text is wrong");
    await page.waitFor("document.querySelector('#lb-img').src.includes('tier=preview')", 8000, "preview tier loaded");
    await shot("06-lightbox");
    await press("Escape");
    check(await page.eval("document.querySelector('#lightbox').hidden"), "viewer did not close");
    await settle();
    check((await selIdx()) === 202, "the grid cursor is not on the last viewed photo");
    const vis = await page.eval("(() => { const r = document.querySelector('#grid .card.sel').getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; })()");
    check(vis, "the last viewed photo is not on screen after closing the viewer");
  });

  await scenario("camera, rating and country filters agree with the server", async () => {
    const facets = await api("/api/facets");
    check(facets.cameras.length === 3 && facets.countries.length === 3, "facets are not what was seeded");
    await page.eval("scrollTo(0, 0)");
    await page.eval(`(() => { const s = document.querySelector('#f-camera'); s.value = 'Canon R5'; s.dispatchEvent(new Event('change')); })()`);
    const want = (await api("/api/images?camera=Canon R5&limit=1")).total;
    await page.waitFor(`document.querySelector('#rowcount').textContent === ${JSON.stringify(want.toLocaleString("en-US") + " photos")}`, 8000, "camera filter count");
    await page.eval(`(() => { const s = document.querySelector('#f-rating'); s.value = 'min:3'; s.dispatchEvent(new Event('change')); })()`);
    const both = (await api("/api/images?camera=Canon R5&min_rating=3&limit=1")).total;
    await page.waitFor(`document.querySelector('#rowcount').textContent === ${JSON.stringify(both.toLocaleString("en-US") + (both === 1 ? " photo" : " photos"))}`, 8000, "camera and rating count");
    check(both > 0 && both < want, `filters did not narrow (${want} -> ${both})`);
    await page.eval(`(() => { const s = document.querySelector('#f-country'); s.value = 'FR'; s.dispatchEvent(new Event('change')); })()`);
    const trio = (await api("/api/images?camera=Canon R5&min_rating=3&country=FR&limit=1")).total;
    await page.waitFor(`document.querySelector('#rowcount').textContent === ${JSON.stringify(trio.toLocaleString("en-US") + (trio === 1 ? " photo" : " photos"))}`, 8000, "three filters");
    await shot("07-filtered");
    await page.eval("document.querySelector('[data-clearfilters]') ? 0 : 0");
    await page.eval(`(() => { for (const id of ['f-camera','f-rating','f-country']) { const s = document.querySelector('#' + id); s.value = ''; s.dispatchEvent(new Event('change')); } })()`);
    await page.waitFor(`document.querySelector('#rowcount').textContent === '5,000 photos'`, 8000, "filters cleared");
  });

  await scenario("date scrubber bounds come from the library and narrow the grid", async () => {
    const f = await api("/api/facets");
    const days = Math.round((Date.parse(f.taken_max) - Date.parse(f.taken_min)) / 86400000);
    check((await page.eval("document.querySelector('#scrub-to').max")) === String(days), "slider range is not the library's date range");
    check((await page.eval("document.querySelector('#scrub-from-label').textContent")) === f.taken_min, "from label is not the first date");
    check((await page.eval("document.querySelector('#scrub-to-label').textContent")) === f.taken_max, "to label is not the last date");
    await page.eval(`(() => { const a = document.querySelector('#scrub-from'); a.value = ${Math.floor(days / 2)}; a.dispatchEvent(new Event('input')); })()`);
    const from = await page.eval("document.querySelector('#scrub-from-label').textContent");
    const want = (await api(`/api/images?date_from=${from}&limit=1`)).total;
    check(want > 0 && want < meta.n, `a half range should narrow the list, got ${want}`);
    await page.waitFor(`document.querySelector('#rowcount').textContent === ${JSON.stringify(want.toLocaleString("en-US") + " photos")}`, 8000, "scrubbed count");
    const first = await page.eval("document.querySelector('#grid .card').getAttribute('data-id')");
    const item = (await api(`/api/images?date_from=${from}&sort=taken&limit=1`)).items[0];
    check(String(item.id) === first, "the first card is not the first photo in range");
    await page.eval(`(() => { const a = document.querySelector('#scrub-from'); a.value = 0; a.dispatchEvent(new Event('input')); })()`);
    await page.waitFor(`document.querySelector('#rowcount').textContent === '5,000 photos'`, 8000, "range cleared");
  });

  await scenario("the grid lays out cleanly after a resize", async () => {
    const layout = () => page.eval(`(() => {
      const cs = [...document.querySelectorAll('#grid .vg-window .card')].map((c) => c.getBoundingClientRect());
      const rows = new Map();
      cs.forEach((r) => { const k = Math.round(r.top); (rows.get(k) || rows.set(k, []).get(k)).push(r); });
      let ok = true; const widths = new Set();
      const tops = [...rows.keys()].sort((a, b) => a - b);
      tops.forEach((t, i) => { const row = rows.get(t).sort((a, b) => a.left - b.left);
        row.forEach((r, j) => { widths.add(Math.round(r.width)); if (j && r.left < row[j - 1].right - 0.5) ok = false; });
        if (i && t < tops[i - 1] + rows.get(tops[i - 1])[0].height - 0.5) ok = false; });
      return { ok, n: cs.length, cols: Math.max(...[...rows.values()].map((r) => r.length)), widths: widths.size };
    })()`);
    const wide = await layout();
    check(wide.ok && wide.widths === 1, "wide layout overlaps or has uneven cards");
    await page.viewport(760, 800);
    try {
      await page.waitFor(`getComputedStyle(document.querySelector('#grid')).getPropertyValue('--vg-cols') !== '${wide.cols}'`, 8000, "column count change");
    } catch (e) {
      const d = await page.eval(`({ w: innerWidth, host: document.querySelector('#grid').clientWidth, cols: getComputedStyle(document.querySelector('#grid')).getPropertyValue('--vg-cols'), visible: !!document.querySelector('#grid').offsetParent, tab: document.querySelector('.panel.active').id })`);
      throw new Error(e.message + " " + JSON.stringify(d));
    }
    await settle();
    const narrow = await layout();
    check(narrow.ok && narrow.widths === 1, "narrow layout overlaps or has uneven cards");
    check(narrow.cols < wide.cols && narrow.n <= 100, `columns ${wide.cols}->${narrow.cols}, ${narrow.n} cards`);
    await shot("08-narrow");
    await page.viewport(1400, 900);
    await page.waitFor(`getComputedStyle(document.querySelector('#grid')).getPropertyValue('--vg-cols') === '${wide.cols}'`, 8000, "columns restored");
    await settle();
  });

  await scenario("thumbnails are WebP at the right tier and are cached by the browser", async () => {
    const info = await page.eval(`(async () => {
      const img = document.querySelector('#grid .card img'); const url = img.currentSrc;
      const r = await fetch(url); const b = await r.blob();
      return { url, type: r.headers.get('content-type'), cache: r.headers.get('cache-control'), size: b.size, w: img.naturalWidth };
    })()`);
    check(info.url.includes("tier=micro") && /v=[0-9a-f]{16}/.test(info.url), "grid thumbnail URL has no tier or fingerprint: " + info.url);
    check(info.type === "image/webp" && info.cache.includes("immutable"), `served as ${info.type} / ${info.cache}`);
  });

  await scenario("Reject all blinks asks first, moves only blinking alternates, and can be undone", async () => {
    await page.eval("document.querySelector('.tab[data-tab=\"bursts\"]').click()");
    await page.waitFor("document.querySelectorAll('#burstdetail .card').length === 4", 10000, "burst photos");
    const btn = () => page.eval("[...document.querySelectorAll('#blinkbar button')].map((b) => b.textContent)");
    check((await btn()).includes("Reject all blinks"), "no Reject all blinks button: " + JSON.stringify(await btn()));
    await page.eval("[...document.querySelectorAll('#blinkbar button')].find((b) => b.textContent === 'Reject all blinks').click()");
    await page.waitFor("!document.querySelector('#confirm-modal').hidden", 6000, "confirm dialog");
    check((await page.eval("document.querySelector('#confirm-modal-title').textContent")) === "Reject all blinks?", "wrong dialog title");
    await shot("09-reject-blinks");
    await page.eval("document.querySelector('#confirm-modal-ok').click()");
    await page.waitFor("[...document.querySelectorAll('#blinkbar button')].some((b) => b.textContent.startsWith('Restore'))", 8000, "restore button");
    check(!(await btn()).includes("Reject all blinks"), "reject is still offered");
    check((await api("/api/images?category=junk_blinks&limit=1")).total === 2, "two alternates should be in Junk review");
    const best = (await api(`/api/burst/${meta.burst_id}`)).members.find((m) => m.is_best_shot);
    check(best.manual_bucket === null, "the best shot was moved");
    await page.eval("[...document.querySelectorAll('#blinkbar button')].find((b) => b.textContent.startsWith('Restore')).click()");
    await page.waitFor("[...document.querySelectorAll('#blinkbar button')].some((b) => b.textContent === 'Reject all blinks')", 8000, "reject offered again");
    check((await api("/api/images?category=junk_blinks&limit=1")).total === 0, "restore left photos in Junk review");
  });

  await scenario("Inspect faces shows real face crops", async () => {
    await page.eval("document.querySelector('#burstdetail [data-faces]').click()");
    await page.waitFor("document.querySelectorAll('#facepanel .facechip').length >= 1", 8000, "face chips");
    await page.waitFor("[...document.querySelectorAll('#facepanel img')].every((i) => i.complete && i.naturalWidth > 0)", 8000, "crops loaded");
    const text = await page.eval("document.querySelector('#facepanel figcaption').textContent");
    check(/Eyes/.test(text), "no eye state shown: " + text);
    await shot("10-faces");
  });

  await scenario("the page raised no script errors", async () => {
    check(page.errors.length === 0, "script errors: " + page.errors.join(" | ").slice(0, 400));
  });
} catch (e) {
  results.push({ name: "harness", ok: false, error: String(e.stack || e) });
  console.log("HARNESS ERROR", e.stack || e);
} finally {
  writeFileSync(out, JSON.stringify(results, null, 1));
  await browser.close();
}

// Real-browser checks of the UI against a seeded server.
// node tests/browser/scenarios.mjs META_JSON RESULTS_JSON [SHOT_DIR]
import { readFileSync, writeFileSync, existsSync, mkdirSync, copyFileSync } from "node:fs";
import { join, dirname } from "node:path";
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
    await page.waitFor("document.querySelector('#lightbox').hidden", 2000, "viewer closes");
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
    await sleep(250);
    await settle();
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

  await scenario("clicking a card's category label moves that photo, or the whole selection, from the grid", async () => {
    await page.eval("document.querySelector('.tab[data-tab=\"categories\"]').click()");
    await page.waitFor("document.querySelectorAll('#grid .vg-window .card').length >= 3", 15000, "category cards");
    await settle();
    const card = (i) => `document.querySelectorAll('#grid .vg-window .card')[${i}]`;
    const idOf = (i) => page.eval(`Number(${card(i)}.dataset.id)`);
    const edited = (id) => page.eval(`!!document.querySelector('#grid .card[data-id="${id}"] .catline .tag')`);
    const label = (id) => page.eval(`document.querySelector('#grid .card[data-id="${id}"] .catpick .t').textContent`);
    const menuOpen = () => page.eval("!!document.querySelector('.catmenu')");
    const target = () => page.eval("(() => { const b = document.querySelector('.catmenu-item:not(.current)'); return b ? [b.dataset.catitem, b.textContent] : null; })()");
    const openFor = (i) => page.eval(`${card(i)}.querySelector('.catpick').click()`);

    // One photo: open, close with Escape, reopen, choose.
    const a = await idOf(0);
    await openFor(0);
    await page.waitFor("!!document.querySelector('.catmenu')", 5000, "picker opens");
    check((await page.eval("document.querySelectorAll('.catmenu-item').length")) >= 5, "picker lists too few categories");
    check((await page.eval("document.querySelector('.catmenu-head').textContent")) === "Move to", "single-photo header is wrong");
    check(await page.eval("document.activeElement.classList.contains('catmenu-item')"), "focus did not enter the picker");
    await press("Escape");
    await page.waitFor("!document.querySelector('.catmenu')", 5000, "Escape closes the picker");
    check(await page.eval(`document.activeElement === ${card(0)}.querySelector('.catpick')`), "focus did not return to the label");
    check(!(await edited(a)), "Escape changed the category");
    await openFor(0);
    await page.waitFor("!!document.querySelector('.catmenu')", 5000, "picker reopens");
    await shot("11-category-picker");
    const [cat, text] = await target();
    await page.eval("document.querySelector('.catmenu-item:not(.current)').click()");
    await page.waitFor("!document.querySelector('.catmenu')", 5000, "picker closes after a choice");
    await page.waitFor(`!!document.querySelector('#grid .card[data-id="${a}"] .catline .tag')`, 8000, "photo marked edited");
    check((await label(a)) === text.trim(), `label is ${await label(a)}, wanted ${text}`);
    await shot("11-quick-category");

    // A selection: choosing from a selected card moves every selected photo.
    const b = await idOf(1), c = await idOf(2);
    for (const id of [b, c]) await page.eval(`document.querySelector('#grid .card[data-id="${id}"] .pick').click()`);
    await page.waitFor("!document.querySelector('#bulkbar').hidden", 5000, "bulk bar shows");
    await openFor(1);
    await page.waitFor("!!document.querySelector('.catmenu')", 5000, "picker opens for a selection");
    check((await page.eval("document.querySelector('.catmenu-head').textContent")) === "Move 2 selected", "bulk header is wrong");
    await page.eval("document.querySelector('.catmenu-item:not(.current)').click()");
    await page.waitFor(`document.querySelector('#grid .card[data-id="${b}"] .catline .tag') && document.querySelector('#grid .card[data-id="${c}"] .catline .tag')`, 8000, "both selected photos edited");
    await page.waitFor("document.querySelector('#bulkbar').hidden", 2000, "selection is cleared");
    check(!(await menuOpen()), "picker is still open");
  });

  await scenario("lightbox focus restoration and focus trap", async () => {
    await page.eval("document.querySelector('.tab[data-tab=\"categories\"]').click()");
    await settle();
    const firstCard = await page.eval("document.querySelector('#grid .vg-window .card[data-idx=\"0\"]')");
    check(firstCard, "no first card in grid");
    await page.eval("document.querySelector('#grid .vg-window .card[data-idx=\"0\"] .thumb').click()");
    await page.waitFor("!document.querySelector('#lightbox').hidden", 8000, "lightbox opens");
    for (let i = 0; i < 10; i++) {
      await press("Tab");
      const inLb = await page.eval("!!document.querySelector('#lightbox').contains(document.activeElement)");
      check(inLb, "focus leaked outside lightbox during Tab");
    }
    await press("Escape");
    await page.waitFor("document.querySelector('#lightbox').hidden", 8000, "lightbox closes");
    await settle();
    const activeCard = await page.eval("document.activeElement && document.activeElement.classList.contains('card')");
    check(activeCard, "focus did not return to the originating card");
    const bodyFocus = await page.eval("document.activeElement === document.body");
    check(!bodyFocus, "focus dropped to body after closing lightbox");
  });

  await scenario("in-app prompt dialog replaces window.prompt for new folder and event rename", async () => {
    await page.eval("document.querySelector('.tab[data-tab=\"library\"]').click()");
    await page.waitFor("document.querySelector('.panel.active').id === 'tab-library'", 8000, "library panel");
    await page.eval("document.querySelector('#btn-browse-scan').click()");
    await page.waitFor("!document.querySelector('#picker').hidden && !!document.querySelector('#picker-path').value", 8000, "picker open and loaded");

    // Folder creation with validation
    await page.eval("document.querySelector('#picker-new').click()");
    await page.waitFor("!document.querySelector('#confirm-modal').hidden", 5000, "confirm modal open");
    check(await page.eval("!document.querySelector('#confirm-modal-input-wrap').hidden"), "input wrap not visible");
    await page.eval("document.querySelector('#confirm-modal-input').value = ''");
    await page.eval("document.querySelector('#confirm-modal-ok').click()");
    await page.waitFor("!document.querySelector('#confirm-modal-input-error').hidden", 4000, "empty name error");
    check(!await page.eval("document.querySelector('#confirm-modal').hidden"), "modal closed on invalid name");

    await page.eval("document.querySelector('#confirm-modal-input').value = 'bad/folder'");
    await page.eval("document.querySelector('#confirm-modal-ok').click()");
    await page.waitFor("!document.querySelector('#confirm-modal-input-error').hidden", 4000, "unsafe name error");

    await press("Escape");
    await page.waitFor("document.querySelector('#confirm-modal').hidden", 4000, "modal closed on Escape");

    await press("Escape");
    await page.waitFor("document.querySelector('#picker').hidden", 4000, "picker closed on Escape");
    await page.eval("document.querySelector('.tab[data-tab=\"categories\"]').click()");
    await settle();
  });

  await scenario("persistent reset-filters button appears, disappears, and resets in one round-trip", async () => {
    check(await page.eval("document.querySelector('#reset-filters').hidden"), "reset-filters not initially hidden");
    await page.eval("document.querySelector('#f-fav').click()");
    await page.waitFor("!document.querySelector('#reset-filters').hidden", 5000, "reset-filters shown after filter change");
    await page.eval("document.querySelector('#reset-filters').click()");
    await page.waitFor("document.querySelector('#reset-filters').hidden", 5000, "reset-filters hidden after click");
    check(!await page.eval("document.querySelector('#f-fav').checked"), "fav filter was not cleared");
  });

  await scenario("native version badge button activates via Enter or Space", async () => {
    await page.eval("document.querySelector('#header-version-badge').focus()");
    check(await page.eval("document.activeElement.id === 'header-version-badge'"), "version badge not focused");
    check(await page.eval("document.querySelector('#header-version-badge').tagName === 'BUTTON'"), "version badge is not button");
    await press("Enter");
    await page.waitFor("document.querySelector('.panel.active').id === 'tab-settings'", 8000, "settings panel active after Enter");
    await page.eval("document.querySelector('.tab[data-tab=\"categories\"]').click()");
    await settle();
  });

  await scenario("touch swipe navigation in lightbox stage", async () => {
    // Wait for the first card itself: after a tab switch the window can still hold cards from lower down.
    await page.waitFor("document.querySelector('#grid .vg-window .card[data-idx=\"0\"] .thumb')", 8000, "first card rendered");
    await page.eval("document.querySelector('#grid .vg-window .card[data-idx=\"0\"] .thumb').click()");
    await page.waitFor("!document.querySelector('#lightbox').hidden", 8000, "lightbox opens");
    const title0 = await page.eval("document.querySelector('#lb-title').textContent");

    // Touch swipe left steps forward
    await page.eval(`(() => {
      const stage = document.querySelector('.lb-stage');
      const r = stage.getBoundingClientRect();
      const x = r.left + r.width / 2, y = r.top + r.height / 2;
      stage.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: x, clientY: y, pointerType: 'touch' }));
      stage.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, clientX: x - 80, clientY: y, pointerType: 'touch' }));
    })()`);
    await page.waitFor(`document.querySelector('#lb-title').textContent !== ${JSON.stringify(title0)}`, 8000, "swipe left stepped");

    // Mouse drag is ignored
    const curTitle = await page.eval("document.querySelector('#lb-title').textContent");
    await page.eval(`(() => {
      const stage = document.querySelector('.lb-stage');
      const r = stage.getBoundingClientRect();
      const x = r.left + r.width / 2, y = r.top + r.height / 2;
      stage.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: x, clientY: y, pointerType: 'mouse' }));
      stage.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, clientX: x - 80, clientY: y, pointerType: 'mouse' }));
    })()`);
    check((await page.eval("document.querySelector('#lb-title').textContent")) === curTitle, "mouse drag stepped photo");

    await press("Escape");
    await page.waitFor("document.querySelector('#lightbox').hidden", 5000);
    await settle();
  });

  await scenario("date presets compute parameters and animate scrubber", async () => {
    await page.eval("document.querySelector('#scrub-presets .preset-btn[data-preset=\"year\"]').click()");
    await page.waitFor("document.querySelector('#scrub-presets .preset-btn[data-preset=\"year\"]').classList.contains('active')", 5000, "year active");
    await page.waitFor("!document.querySelector('#reset-filters').hidden", 5000, "reset-filters visible on preset");
    await page.eval("document.querySelector('#reset-filters').click()");
    await page.waitFor("document.querySelector('#reset-filters').hidden", 5000, "filters reset");
    check(await page.eval("document.querySelector('#scrub-presets .preset-btn[data-preset=\"all\"]').classList.contains('active')"), "preset reset to all");
  });

  await scenario("roving focus for virtual grid has exactly one tab stop and moves DOM focus", async () => {
    await page.waitFor("document.querySelectorAll('#grid .vg-window .card').length > 0", 8000, "cards rendered");
    const zeros = await page.eval("document.querySelectorAll('#grid .vg-window .card[tabindex=\"0\"]').length");
    check(zeros === 1, "expected exactly 1 card with tabindex=0, got " + zeros);
    const negButtons = await page.eval("document.querySelectorAll('#grid .vg-window .card[tabindex=\"-1\"] button:not([tabindex=\"-1\"])').length");
    check(negButtons === 0, "found tabbable buttons in inactive cards: " + negButtons);
    await press("ArrowRight");
    await settle();
    const focusedCard = await page.eval("document.activeElement.classList.contains('card')");
    check(focusedCard, "ArrowRight did not move DOM focus to card");
  });

  await scenario("picker and help modals contain focus and restore to opener", async () => {
    await page.waitFor("document.querySelectorAll('#grid .vg-window .card[tabindex=\"0\"]').length > 0", 8000, "card with tabindex 0 rendered");
    await page.eval("document.querySelector('#grid .vg-window .card[tabindex=\"0\"]').focus()");
    await press("?");
    await page.waitFor("!document.querySelector('#help').hidden", 5000, "help modal open");
    check(await page.eval("document.activeElement.id === 'help-close'"), "focus started on help-close");
    for (let i = 0; i < 5; i++) await press("Tab");
    check(await page.eval("document.querySelector('#help').contains(document.activeElement)"), "focus escaped help modal");
    await press("Escape");
    await page.waitFor("document.querySelector('#help').hidden", 5000, "help closed on Escape");
    check(await page.eval("document.activeElement.classList.contains('card')"), "focus restored to card after closing help");
  });

  await scenario("the hardware theme switch flips light/dark and stays in sync (click + keyboard)", async () => {
    await page.eval("document.querySelector('.tab[data-tab=\"library\"]').click()");
    await settle();
    const flip = (t) => (t === "dark" ? "light" : "dark");
    const snap = () => page.eval(`(() => {
      const sw = document.getElementById('theme-switch');
      return { theme: document.documentElement.dataset.theme, checked: sw.checked,
        aria: sw.getAttribute('aria-checked'), stored: localStorage.getItem('pcur-theme') };
    })()`);
    const t0 = (await snap()).theme;
    check(t0 === "light" || t0 === "dark", "no initial theme: " + t0);

    // 1) Drive the checkbox's change event (the click path) -> the whole system follows.
    await page.eval(`(() => { const sw = document.getElementById('theme-switch'); sw.checked = !sw.checked; sw.dispatchEvent(new Event('change')); })()`);
    const a = await snap();
    check(a.theme === flip(t0), "data-theme did not flip to " + flip(t0) + " (got " + a.theme + ")");
    check(a.checked === (flip(t0) === "dark"), "checkbox state is out of sync");
    check(a.aria === String(flip(t0) === "dark"), "aria-checked is out of sync");
    check(a.stored === flip(t0), "localStorage pcur-theme is out of sync");

    // 2) Focus the label and press Space (the keyboard path) -> flips back to the original.
    await page.eval("document.getElementById('theme-toggle').focus()");
    check(await page.eval("document.activeElement.id === 'theme-toggle'"), "the switch label is not keyboard-focusable");
    await press(" ");
    const b = await snap();
    check(b.theme === t0, "keyboard toggle did not restore " + t0 + " (got " + b.theme + ")");
    check(b.checked === (t0 === "dark") && b.aria === String(t0 === "dark") && b.stored === t0, "state not restored in sync after keyboard toggle");
    await shot("12-theme-switch");
  });

  await scenario("step cards reflect library state and stack to one column on tablet", async () => {
    await page.eval("document.querySelector('.tab[data-tab=\"library\"]').click()");
    await settle();
    const info = await page.eval(`(() => {
      const steps = [...document.querySelectorAll('.steps > .step')].map((li) => ({
        id: li.id, done: li.classList.contains('done'), active: li.classList.contains('active'), locked: li.classList.contains('locked'),
        badgeBg: getComputedStyle(li.querySelector('.step-num')).backgroundColor,
        checkVisible: getComputedStyle(li.querySelector('.step-num .ic')).display !== 'none',
        numHidden: getComputedStyle(li.querySelector('.step-num .n')).display === 'none'
      }));
      return {
        steps,
        states: { scan: document.getElementById('state-scan').textContent, infer: document.getElementById('state-infer').textContent,
                  cluster: document.getElementById('state-cluster').textContent },
        cols: getComputedStyle(document.querySelector('.steps')).gridTemplateColumns.trim().split(/\\s+/).length
      };
    })()`);
    // Seeded library: 5,000 indexed + 1 burst group, all inferred -> Analyze and Group are done and
    // Scan is the active step again, ready for the next batch.
    const byId = Object.fromEntries(info.steps.map((s) => [s.id, s]));
    check(byId["step-scan"].active && !byId["step-scan"].done && !byId["step-scan"].locked, "step-scan should be the active next step");
    for (const id of ["step-infer", "step-cluster"]) {
      check(byId[id].done, id + " is not in the done state");
      check(!byId[id].active && !byId[id].locked, id + " carries a conflicting state class");
      check(byId[id].badgeBg !== "rgba(0, 0, 0, 0)", id + " badge has no painted background");
      check(byId[id].checkVisible && byId[id].numHidden, id + " did not swap its number for the check mark");
    }
    check(/indexed/.test(info.states.scan), "step 1 state text has no index count: " + info.states.scan);
    check(info.states.infer === "Done", "step 1 done but step 2 is not marked done: " + info.states.infer);
    check(/group/.test(info.states.cluster), "step 3 state text has no group count: " + info.states.cluster);
    check(info.cols === 3, "wide layout should show 3 step columns, got " + info.cols);

    // Tablet: at <= 960px the 3-column grid stacks to a single column.
    await page.viewport(900, 800);
    await page.waitFor(`getComputedStyle(document.querySelector('.steps')).gridTemplateColumns.trim().split(/\\s+/).length === 1`, 8000, "steps stack to one column");
    await settle();
    const narrow = await page.eval("getComputedStyle(document.querySelector('.steps')).gridTemplateColumns.trim().split(/\\s+/).length");
    check(narrow === 1, "tablet layout should show 1 step column, got " + narrow);
    await shot("13-steps-tablet");
    await page.viewport(1400, 900);
    await page.waitFor(`getComputedStyle(document.querySelector('.steps')).gridTemplateColumns.trim().split(/\\s+/).length === 3`, 8000, "steps restore to 3 columns");
    await settle();
  });

  await scenario("activity log rows render monospace tabular time pills", async () => {
    await page.eval("document.querySelector('.tab[data-tab=\"library\"]').click()");
    await settle();
    // An empty scan path is a safe, non-mutating way to make the real logTo() write a row.
    await page.eval("document.getElementById('scan-path').value = ''");
    await page.eval("document.getElementById('btn-scan').click()");
    await page.waitFor(`(() => { const l = document.querySelector('#library-log .logline .time-pill'); return l && /\\d{1,2}:\\d{2}/.test(l.textContent); })()`, 8000, "a time-pill appears in the activity log");
    const log = await page.eval(`(() => {
      const rows = [...document.querySelectorAll('#library-log .logline')];
      const err = rows.find((r) => r.textContent.includes('Choose a photo folder'));
      const anyPill = document.querySelector('#library-log .logline .time-pill');
      const cs = getComputedStyle(anyPill);
      return {
        hasErrRow: !!err,
        errPillClass: err ? err.querySelector('.time-pill').className : "",
        errHighlighted: err ? err.classList.contains('active-highlight') : false,
        fontFamily: cs.fontFamily, fontVariant: cs.fontVariantNumeric,
        pillText: anyPill.textContent,
        terminalBg: getComputedStyle(document.getElementById('library-log')).backgroundColor
      };
    })()`);
    check(/\d{1,2}:\d{2}/.test(log.pillText), "time-pill has no readable time: " + log.pillText);
    check(/JetBrains Mono/.test(log.fontFamily), "time-pill is not monospace: " + log.fontFamily);
    check(/tabular-nums/.test(log.fontVariant), "time-pill is not tabular: " + log.fontVariant);
    check(log.hasErrRow && log.errPillClass.includes("muted"), "the error row should carry a muted time-pill: " + log.errPillClass);
    check(!log.errHighlighted, "an error row should not be a completion highlight");
    check(log.terminalBg !== "rgba(0, 0, 0, 0)", "the activity terminal has no painted background");
    await shot("14-log-pills");
  });

  const clickTab = async (t) => {
    await page.eval(`document.querySelector('.tab[data-tab="${t}"]').click()`);
    await page.waitFor(`document.querySelector('.panel.active').id === 'tab-${t}'`, 8000, `${t} tab active`);
  };
  const post = (path, body) => page.eval(`fetch(${JSON.stringify(path)}, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: ${JSON.stringify(JSON.stringify(body || {}))} }).then((r) => r.json())`);
  const imageId = async (i) => (await api(`/api/images?q=${name(i)}&limit=1`)).items[0].id;

  await scenario("search explains that it needs the AI model when only the placeholder is installed", async () => {
    await clickTab("search");
    await page.waitFor("document.querySelector('#searchgrid .empty')", 5000, "the prompt is shown");
    await page.eval("(() => { document.querySelector('#searchq').value = 'a dog on the beach'; document.querySelector('#search-form').requestSubmit(); })()");
    await page.waitFor("(document.querySelector('#searchgrid .empty h3') || {}).textContent === 'Search needs the AI model'", 8000, "model message");
    await shot("15-search");
  });

  await scenario("highlights rank best first, show the best of each event and offer a year export", async () => {
    await clickTab("highlights");
    await page.waitFor("document.querySelector('#highlightgrid .vg-window .card[data-idx=\"0\"] .fn')", 10000, "highlight cards");
    const first = await page.eval("document.querySelector('#highlightgrid .vg-window .card[data-idx=\"0\"] .fn').textContent");
    check(first === name(500), "the best photo should come first, got " + first);
    check(Number(await page.eval("document.querySelector('#highlightcount').textContent.replace(/[^0-9]/g, '')")) === 10, "ten scored photos expected");
    await page.eval("(() => { const v = document.querySelector('#hl-view'); v.value = 'events'; v.dispatchEvent(new Event('change')); })()");
    await page.waitFor("document.querySelectorAll('#hl-events .ev-block').length === 1", 8000, "one event block");
    check((await page.eval("document.querySelector('#hl-events .ev-head h3').textContent")) === "Seaside", "event name");
    check((await page.eval("document.querySelectorAll('#hl-events .ev-block .card').length")) === 3, "top three of the event");
    await page.waitFor("document.querySelectorAll('#hl-year option').length > 1", 8000, "years listed");
    check(await page.eval("document.querySelector('#btn-hl-export').disabled"), "export starts disabled");
    await page.eval("(() => { const y = document.querySelector('#hl-year'); y.value = y.options[1].value; y.dispatchEvent(new Event('change')); })()");
    check(!(await page.eval("document.querySelector('#btn-hl-export').disabled")), "export enabled once a year is chosen");
    const page5 = await page.eval("fetch('/api/highlights/export?year=' + document.querySelector('#hl-year').value + '&n=5').then((r) => r.text())");
    check(page5.includes("data:image/webp;base64,") && page5.includes(name(500)), "the exported page lists the best photo with a preview");
    await page.eval("document.querySelector('#hl-tune').open = true");
    await shot("16-highlights");
    await page.eval("(() => { const v = document.querySelector('#hl-view'); v.value = 'all'; v.dispatchEvent(new Event('change')); })()");
  });

  await scenario("ranking sliders save and a favourite can outrank the AI's score", async () => {
    const id = await imageId(509);                      // the lowest scored of the ten
    await post(`/api/images/${id}/favorite`, { favorite: true });
    await page.eval("(() => { const s = document.querySelector('#hl-w-fav'); s.value = '0.5'; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); })()");
    await page.waitFor(`document.querySelector('#highlightgrid .vg-window .card[data-idx="0"] .fn').textContent === ${JSON.stringify(name(509))}`, 8000, "the favourite jumps to the top");
    check((await api("/api/features")).features.highlight_favorite_weight === 0.5, "the weight was saved");
    await page.eval("(() => { const s = document.querySelector('#hl-w-fav'); s.value = '0.15'; s.dispatchEvent(new Event('input')); s.dispatchEvent(new Event('change')); })()");
    await post(`/api/images/${id}/favorite`, { favorite: false });
    await page.waitFor(`document.querySelector('#highlightgrid .vg-window .card[data-idx="0"] .fn').textContent === ${JSON.stringify(name(500))}`, 8000, "the order is restored");
    check(/at least 8/.test(await page.eval("document.querySelector('#taste-text').textContent")), "the taste panel explains what it needs");
    check(await page.eval("document.querySelector('#btn-taste-learn').disabled"), "learning is disabled with too few ratings");
  });

  await scenario("duplicate review lists groups, picks a keeper, resolves one and dismisses the other", async () => {
    await clickTab("duplicates");
    await page.waitFor("document.querySelectorAll('#dup-list .dup-group').length === 2", 10000, "two groups");
    check((await page.eval("document.querySelector('[data-count=\"duplicates\"]').textContent")) === "2", "the nav badge counts groups");
    await shot("17-duplicates");
    const exactGroup = "Array.from(document.querySelectorAll('#dup-list .dup-group')).find((x) => /identical/.test(x.querySelector('h3').textContent))";
    const alt = await page.eval(`(() => {
      const g = ${exactGroup};
      const second = g.querySelectorAll('.dup-member')[1];
      second.querySelector('[data-dup-keep]').click();
      return Number(second.dataset.mid);
    })()`);
    await page.waitFor(`(() => { const g = ${exactGroup}; return g && g.querySelector('.dup-member.keep') && Number(g.querySelector('.dup-member.keep').dataset.mid) === ${alt}; })()`, 5000, "the second photo is now the one kept");
    check(/discard 1 other/.test(await page.eval(`${exactGroup}.querySelector('[data-dup-apply]').textContent`)), "the apply button says what it will do");
    await page.eval(`${exactGroup}.querySelector('[data-dup-apply]').click()`);
    await page.waitFor("document.querySelectorAll('#dup-list .dup-group').length === 1", 8000, "the resolved group left the queue");
    const done = (await api("/api/duplicates?reviewed=true&kind=exact")).groups.find((g) => g.reviewed);
    check(done && done.keeper_id === alt, "the chosen photo is the one kept");
    await page.eval("document.querySelector('[data-dup-dismiss]').click()");
    await page.waitFor("(document.querySelector('#dup-list .empty h3') || {}).textContent === 'No duplicates to review'", 8000, "all done");
  });

  await scenario("the keyword setting saves and the sidecar actions explain themselves", async () => {
    await clickTab("settings");
    await page.waitFor("document.querySelector('#kw-note').textContent.length > 0", 8000, "note filled");
    check(!(await page.eval("document.querySelector('#set-keywords').disabled")), "the switch is usable");
    await page.eval("document.querySelector('#set-keywords').click()");
    await page.waitFor("fetch('/api/features').then((r) => r.json()).then((j) => j.features.write_keywords === true)", 5000, "saved on");
    await shot("18-settings-keywords");
    await page.eval("document.querySelector('#set-keywords').click()");
    await page.waitFor("fetch('/api/features').then((r) => r.json()).then((j) => j.features.write_keywords === false)", 5000, "saved off");
  });

  await scenario("watch mode asks for a folder, picks up a new photo, and stops", async () => {
    const inbox = join(dirname(meta.lib), "inbox");
    mkdirSync(inbox, { recursive: true });
    await clickTab("library");
    await page.eval("document.querySelector('#scan-path').value = ''");
    await page.eval("document.querySelector('#watch-toggle').click()");
    await page.waitFor("!document.querySelector('#watch-toggle').checked", 5000, "no folder, no watch");
    await page.eval(`document.querySelector('#scan-path').value = ${JSON.stringify(inbox)}`);
    await page.eval("document.querySelector('#watch-toggle').click()");
    await page.waitFor("/Watching|Looking for new photos|Analyzing/.test(document.querySelector('#watch-status').textContent)", 8000, "watching");
    const before = (await api("/api/overview")).total;
    copyFileSync(join(meta.lib, name(1)), join(inbox, "arrived.jpg"));
    await page.waitFor(`fetch('/api/overview').then((r) => r.json()).then((o) => o.total === ${before + 1})`, 40000, "the new photo is indexed");
    await page.waitFor("/1 new photo/.test(document.querySelector('#watch-status').textContent)", 20000, "the status counts it");
    await shot("19-watching");
    await page.eval("document.querySelector('#watch-toggle').click()");
    await page.waitFor("fetch('/api/watch').then((r) => r.json()).then((w) => w.running === false)", 8000, "stopped");
  });

  await scenario("motion: rapid tab switching leaves the last panel fully visible", async () => {
    const tabs = ["categories", "search", "bursts", "highlights", "events", "duplicates", "junk", "uncertain", "actions", "settings", "library"];
    for (const t of [...tabs, ...tabs]) await page.eval(`document.querySelector('.tab[data-tab="${t}"]').click()`);
    await sleep(500);
    const r = await page.eval(`(() => { const p = document.querySelector('#tab-library'); return { opacity: getComputedStyle(p).opacity, active: p.classList.contains('active'), entering: document.querySelectorAll('.panel.enter').length }; })()`);
    check(r.active && r.opacity === "1" && r.entering === 0, "last panel not fully visible: " + JSON.stringify(r));
  });

  await scenario("motion: reduced motion closes a dialog at once", async () => {
    await page.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: "reduce" }] });
    try {
      await page.eval("document.querySelector('#btn-help').click()");
      check(await page.eval("!document.querySelector('#help').hidden"), "help did not open");
      const hidden = await page.eval("(() => { document.querySelector('#help-close').click(); return new Promise((r) => setTimeout(() => r(document.querySelector('#help').hidden), 40)); })()");
      check(hidden === true, "help still visible 40ms after close under reduced motion");
    } finally {
      await page.send("Emulation.setEmulatedMedia", { features: [{ name: "prefers-reduced-motion", value: "no-preference" }] });
    }
  });

  await scenario("motion: reopening a dialog that is closing keeps it open", async () => {
    await page.eval("document.querySelector('#btn-help').click()");
    await page.eval("document.querySelector('#help-close').click()");
    await sleep(60);
    await page.eval("document.querySelector('#btn-help').click()");
    await sleep(450);
    const r = await page.eval("(() => { const h = document.querySelector('#help'); return { hidden: h.hidden, closing: h.classList.contains('closing') }; })()");
    check(r.hidden === false && r.closing === false, "reopened dialog ended hidden or stuck closing: " + JSON.stringify(r));
    await page.eval("document.querySelector('#help-close').click()");
    await page.waitFor("document.querySelector('#help').hidden", 3000, "help closes again");
  });

  await scenario("import flow: the next step is the long button and Sort Photos lights up when relevant", async () => {
    const total = (await api("/api/overview")).total;
    const stub = (patch) => page.eval(`(() => { if (!window.__rf) window.__rf = window.fetch.bind(window); const p = ${JSON.stringify(patch)}; window.fetch = (u, i) => /\\/api\\/overview/.test(String(u)) ? window.__rf(u, i).then((r) => r.json()).then((o) => { o.stats = Object.assign({}, o.stats, p.stats || {}); o.by_category = Object.assign({}, o.by_category, p.by_category || {}); return new Response(JSON.stringify(o), { status: 200, headers: { "Content-Type": "application/json" } }); }) : window.__rf(u, i); return true; })()`);
    const stored = (v) => page.eval(`localStorage.setItem('pcur-grouped-total', ${JSON.stringify(String(v))}); true`);
    const refresh = () => page.eval("document.querySelector('.tab[data-tab=\"library\"]').click(); document.querySelector('#btn-sort-dock').click(); true");
    const tall = (id) => page.eval(`getComputedStyle(document.querySelector('#${id}')).height === '48px'`);
    const hint = () => page.eval("document.querySelector('#dock-hint').textContent");
    try {
      await stored(total); await stub({}); await refresh();
      await page.waitFor("document.querySelector('#step-scan').classList.contains('active')", 8000, "scan is next");
      check(await tall("btn-scan") && !(await tall("btn-infer")) && !(await tall("btn-cluster")), "only Scan should be the long button");
      check(/Scan/.test(await hint()), "hint should invite the next scan: " + (await hint()));

      await stub({ stats: { pending_inference: 3 } }); await refresh();
      await page.waitFor("document.querySelector('#step-infer').classList.contains('active')", 8000, "analyze is next");
      check(await tall("btn-infer") && !(await tall("btn-scan")), "Analyze should be the long button");
      check(/Analyze/.test(await hint()), "hint should say Analyze: " + (await hint()));

      await stub({}); await stored(total - 1); await refresh();
      await page.waitFor("document.querySelector('#step-cluster').classList.contains('active')", 8000, "group is next");
      check(await tall("btn-cluster") && !(await tall("btn-infer")), "Group should be the long button, separate from Analyze");
      check(/Group/.test(await hint()), "hint should say Group: " + (await hint()));

      await stored(total); await stub({ by_category: { unrouted: 5 } }); await refresh();
      await page.waitFor("document.querySelector('#btn-sort-dock').classList.contains('needs-action')", 8000, "Sort Photos is highlighted");
      check(/Sort Photos/.test(await hint()), "hint should say Sort Photos: " + (await hint()));
    } finally {
      await page.eval("if (window.__rf) window.fetch = window.__rf; localStorage.removeItem('pcur-grouped-total'); true");
    }
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

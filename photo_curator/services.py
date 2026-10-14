"""High-level orchestration services shared by the CLI and the web UI.

Ties together the clustering, decision and organizer steps to produce a
complete, reviewable :class:`Plan`, and renders the dry-run audit report.
"""
from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table

from .clustering.best_shot_selector import BestShotSelector, _candidate_from_record
from .clustering.burst_detector import BurstDetector
from .config import CuratorConfig
from .db.models import BurstGroup, ImageRecord
from .db.repository import PhotoRepository
from .db.session import get_session
from .engine.action_engine import ActionEngine, ExecutionReport
from .engine.rollback import RollbackEngine, RollbackReport
from .rules.decision_matrix import DecisionMatrix
from .rules.organizer import FolderOrganizer, OrganizeMode, PlannedAction
from .rules.taxonomy import CategoryType

console = Console()

_EPOCH = datetime(1970, 1, 1)


@dataclass
class Plan:
    plan_id: str
    mode: str
    operation: str
    organize_root: str
    created_at: float
    total: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    actions: List[PlannedAction] = field(default_factory=list)
    burst_count: int = 0
    dry_run: bool = True
    report_path: Optional[str] = None


class Planner:
    """Cluster + decide + organize into a reviewable plan."""

    def __init__(self, config: CuratorConfig | None = None,
                 db_path: Optional[Path] = None):
        self.cfg = config or CuratorConfig()
        self.db_path = db_path or self.cfg.db_path

    def build_plan(self,
                   organize_root: Optional[Path] = None,
                   operation: str = "move",
                   mode: OrganizeMode = OrganizeMode.CATEGORY_FIRST,
                   plan_id: Optional[str] = None,
                   include_duplicates: bool = True) -> Plan:
        root = Path(organize_root or self.cfg.organize_root)
        pid = plan_id or f"PLAN_{int(time.time())}"
        with get_session(self.db_path) as session:
            repo = PhotoRepository(session)
            records = repo.all_images()

            # ---- reset previous burst state
            for r in records:
                r.burst_id = None
                r.is_best_shot = False
                r.is_alternate = False
            for b in repo.all_bursts():
                repo.s.delete(b)
            session.flush()

            # ---- clustering
            detector = BurstDetector(self.cfg)
            burst_results = detector.cluster_records(records)
            by_id = {r.id: r for r in records}

            burst_count = 0
            for res in burst_results:
                members = [by_id[m] for m in res.image_ids if m in by_id]
                if len(members) < self.cfg.burst.min_group_size:
                    continue
                best_id = BestShotSelector(self.cfg).pick_best_shot(
                    [_candidate_from_record(m) for m in members]).image_id
                earliest = min(members, key=lambda m: (m.taken_at is None,
                                                       m.taken_at or _EPOCH))
                stamp = earliest.taken_at.strftime("%Y%m%d") if earliest.taken_at \
                    else "Unknown"
                burst_name = f"Burst_{stamp}_{burst_count:04d}"
                bg = repo.upsert_burst(BurstGroup(
                    name=burst_name, image_count=len(members),
                    best_shot_id=best_id,
                    start_time=earliest.taken_at,
                    end_time=max((m.taken_at for m in members), default=earliest.taken_at),
                ))
                burst_count += 1
                for m in members:
                    m.burst_id = bg.id
                    if m.id == best_id:
                        m.is_best_shot = True
                    else:
                        m.is_alternate = True

            # ---- decision + organize
            matrix = DecisionMatrix(self.cfg)
            organizer = FolderOrganizer(self.cfg, organize_root=root, mode=mode)
            organizer.set_bursts(repo.all_bursts())

            actions: List[PlannedAction] = []
            by_category: Dict[str, int] = {}
            for r in records:
                if r.infer_status != "inferred":
                    continue
                if r.is_duplicate and not include_duplicates:
                    continue
                dec = matrix.evaluate_record(r)
                target = organizer.full_target(r, dec)
                op = operation
                if mode == OrganizeMode.IN_PLACE and op == "move":
                    op = "hardlink"
                actions.append(PlannedAction(
                    image_id=r.id, source=Path(r.path), target=target,
                    operation=op, category=dec.category, reason=dec.reason,
                    is_alternate=dec.is_alternate,
                    burst_name=organizer.burst_name_for(r) if dec.is_alternate else None,
                ))
                by_category[dec.category.value] = \
                    by_category.get(dec.category.value, 0) + 1
                repo.set_routing(r, dec.category.value, dec.subcategory,
                                 str(target), route_status="planned")

        return Plan(
            plan_id=pid, mode=mode.value, operation=operation,
            organize_root=str(root), created_at=time.time(), total=len(actions),
            by_category=by_category, actions=actions, burst_count=burst_count,
        )

    # -------------------------------------------------------------- execute
    def execute_plan(self, plan: Plan,
                     operation: Optional[str] = None) -> ExecutionReport:
        engine = ActionEngine(self.cfg, db_path=self.db_path)
        if operation:
            for a in plan.actions:
                a.operation = operation
        return engine.execute(plan.actions)

    def dry_run(self, plan: Plan) -> ExecutionReport:
        return ActionEngine(self.cfg, db_path=self.db_path).dry_run(plan.actions)

    def undo(self, transaction_id: str) -> RollbackReport:
        return RollbackEngine(self.cfg, db_path=self.db_path).rollback_transaction(
            transaction_id)

    # --------------------------------------------------------------- report
    def render_html_report(self, plan: Plan, output: Path | str,
                           extra: Optional[dict] = None) -> Path:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = _plan_to_rows(self.db_path, plan)
        out.write_text(_render_html(plan, rows, extra or {}), encoding="utf-8")
        return out

    def print_summary(self, plan: Plan) -> None:
        t = Table(title=f"Plan {plan.plan_id}  ({plan.mode} / {plan.operation})")
        t.add_column("Category", style="cyan")
        t.add_column("Count", justify="right")
        t.add_column("Target folder", style="dim")
        for cat, n in sorted(plan.by_category.items(), key=lambda kv: -kv[1]):
            try:
                folder = CategoryType(cat).folder
            except ValueError:
                folder = cat
            t.add_row(cat, str(n), folder)
        t.add_row("[bold]Total[/bold]", f"[bold]{plan.total}[/bold]",
                  f"{plan.burst_count} burst groups")
        console.print(t)
        console.print(f"[dim]Organize root: {plan.organize_root}[/dim]")


def _plan_to_rows(db_path: Path, plan: Plan) -> List[dict]:
    with get_session(db_path) as session:
        repo = PhotoRepository(session)
        rows = []
        for a in plan.actions:
            img = repo.get_by_id(a.image_id)
            if img is None:
                continue
            sem = getattr(img, "semantic", None)
            q = getattr(img, "quality", None)
            rows.append({
                "image_id": a.image_id,
                "filename": img.filename,
                "relative_path": img.relative_path,
                "category": a.category.value,
                "reason": a.reason,
                "target": str(a.target),
                "operation": a.operation,
                "is_alternate": a.is_alternate,
                "is_duplicate": img.is_duplicate,
                "taken_at": img.taken_at.isoformat() if img.taken_at else "",
                "width": img.width, "height": img.height,
                "top1": sem.top1_category if sem else "",
                "top1_score": round(sem.top1_score, 3) if sem else 0,
                "margin": round(sem.score_margin, 3) if sem else 0,
                "quality_score": round(q.quality_score, 3) if q else 0,
                "is_blurry": bool(q.is_blurry) if q else False,
                "is_too_dark": bool(q.is_too_dark) if q else False,
                "is_overexposed": bool(q.is_overexposed) if q else False,
                "has_face": bool(getattr(img, "faces", None)),
            })
        return rows


def _rel_target(target: str, root: str) -> str:
    try:
        return str(Path(target).relative_to(root))
    except ValueError:
        return target


def _render_html(plan: Plan, rows: List[dict], extra: dict) -> str:
    total = plan.total
    cats = plan.by_category

    def pct(n):
        return f"{(100.0 * n / total):.1f}%" if total else "0.0%"

    cat_cards = "".join(
        f'<div class="card"><div class="card-n">{v}</div>'
        f'<div class="card-l">{html.escape(c)}</div>'
        f'<div class="card-p">{pct(v)}</div></div>'
        for c, v in sorted(cats.items(), key=lambda kv: -kv[1])
    )

    trs = []
    for r in rows:
        badges = []
        if r["is_duplicate"]:
            badges.append('<span class="b dup">DUP</span>')
        if r["is_alternate"]:
            badges.append('<span class="b alt">ALT</span>')
        if r["has_face"]:
            badges.append('<span class="b face">face</span>')
        if r["is_blurry"]:
            badges.append('<span class="b blur">blurry</span>')
        if r["is_too_dark"]:
            badges.append('<span class="b dark">dark</span>')
        if r["is_overexposed"]:
            badges.append('<span class="b over">over</span>')
        taken = r["taken_at"][:19].replace("T", " ") if r["taken_at"] else "—"
        clip = (f'{html.escape(r["top1"])} {r["top1_score"]}' if r["top1"] else "—")
        trs.append(
            "<tr>"
            f'<td class="fn" title="{html.escape(r["relative_path"])}">{html.escape(r["filename"])}</td>'
            f'<td class="cat" data-cat="{html.escape(r["category"])}">{html.escape(r["category"])}</td>'
            f'<td>{"".join(badges)}</td>'
            f'<td class="num">{r["width"]}×{r["height"]}</td>'
            f'<td class="num">{clip}</td>'
            f'<td class="num">{r["quality_score"]}</td>'
            f'<td class="date">{taken}</td>'
            f'<td class="reason" title="{html.escape(r["reason"])}">{html.escape(r["reason"][:60])}</td>'
            f'<td class="target" title="{html.escape(r["target"])}">{html.escape(_rel_target(r["target"], plan.organize_root))}</td>'
            f'<td class="op">{html.escape(r["operation"])}</td>'
            "</tr>"
        )

    meta_html = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in extra.items()
    )
    cats_json = json_dumps(cats)  # raw JSON; keys are CategoryType values (safe)

    return _HTML_TEMPLATE.format(
        plan_id=html.escape(plan.plan_id),
        mode=html.escape(plan.mode),
        operation=html.escape(plan.operation),
        total=total,
        burst_count=plan.burst_count,
        dupes=sum(1 for r in rows if r["is_duplicate"]),
        ncats=len(cats),
        meta=meta_html,
        cards=cat_cards,
        rows="".join(trs),
        cats_json=cats_json,
        root=html.escape(plan.organize_root),
    )


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj)


# ``{}`` in the JS below are escaped as ``{{ }}`` for str.format.
_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photo Curator — Plan {plan_id}</title>
<style>
:root{{--bg:#0f1117;--panel:#171a23;--panel2:#1e2230;--fg:#e7e9f0;--mut:#8b93a7;
--acc:#5b8cff;--line:#262b3a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
header{{padding:22px 26px;border-bottom:1px solid var(--line);
display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px}}
h1{{margin:0;font-size:20px}}
.sub{{color:var(--mut);font-size:13px;margin-top:4px}}
.stats{{display:flex;gap:10px;flex-wrap:wrap}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;min-width:96px}}
.stat b{{display:block;font-size:20px}}
.stat span{{color:var(--mut);font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;padding:18px 26px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}}
.card-n{{font-size:26px;font-weight:600;color:var(--acc)}}
.card-l{{text-transform:capitalize}}
.card-p{{color:var(--mut);font-size:12px}}
.controls{{display:flex;gap:10px;padding:0 26px 8px;flex-wrap:wrap;align-items:center}}
.controls input,.controls select{{background:var(--panel2);border:1px solid var(--line);
color:var(--fg);border-radius:8px;padding:8px 10px;font-size:13px}}
.controls input{{min-width:260px}}
table{{width:calc(100% - 52px);margin:0 26px 12px;border-collapse:collapse;font-size:13px}}
th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}
th{{position:sticky;top:0;background:var(--panel);color:var(--mut);font-weight:600;cursor:pointer}}
td.num{{font-variant-numeric:tabular-nums;color:var(--mut);white-space:nowrap}}
td.date{{white-space:nowrap;color:var(--mut)}}
td.fn{{max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
td.target{{max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--acc)}}
td.reason{{max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--mut)}}
tr:hover td{{background:rgba(91,140,255,.06)}}
.b{{display:inline-block;font-size:10px;padding:2px 6px;border-radius:6px;margin-right:4px;font-weight:600}}
.b.dup{{background:#3a2530;color:#ff9db4}}
.b.alt{{background:#25324a;color:#8ab4ff}}
.b.face{{background:#1f3a2b;color:#7fe0a6}}
.b.blur{{background:#3a3320;color:#f2d17a}}
.b.dark{{background:#2b2f3a;color:#aab4cf}}
.b.over{{background:#3a2a24;color:#f2b48a}}
.cat{{text-transform:capitalize}}
table.meta{{width:calc(100% - 52px);margin:0 26px 10px;border-collapse:collapse}}
table.meta th{{text-align:left;color:var(--mut);font-weight:500;width:200px;padding:4px 8px}}
table.meta td{{padding:4px 8px;color:var(--fg)}}
footer{{padding:0 26px 30px;color:var(--mut);font-size:12px}}
</style>
</head><body>
<header>
  <div>
    <h1>Photo Curator — Dry-Run Plan</h1>
    <div class="sub">Plan <b>{plan_id}</b> · mode <b>{mode}</b> · operation <b>{operation}</b></div>
  </div>
  <div class="stats">
    <div class="stat"><b>{total}</b><span>files to route</span></div>
    <div class="stat"><b>{burst_count}</b><span>burst groups</span></div>
    <div class="stat"><b>{dupes}</b><span>duplicates</span></div>
    <div class="stat"><b>{ncats}</b><span>categories</span></div>
  </div>
</header>
<table class="meta"><tbody>{meta}</tbody></table>
<div class="cards">{cards}</div>
<div class="controls">
  <input id="q" type="search" placeholder="Search filename, path or reason…">
  <select id="cat"><option value="">All categories</option></select>
  <span id="count" style="color:var(--mut)"></span>
</div>
<table id="tbl">
<thead><tr>
<th data-k="0">File</th><th data-k="1">Category</th><th>Flags</th>
<th data-k="3">Size</th><th data-k="4">CLIP</th>
<th data-k="5">Quality</th><th data-k="6">Taken</th>
<th>Reason</th><th>Target</th><th>Op</th></tr></thead>
<tbody id="rows">{rows}</tbody></table>
<footer>Generated by <b>photo-curator</b> · Organize root:
<code>{root}</code> · Dry-run: nothing has been moved.</footer>
<script id="cats" type="application/json">{cats_json}</script>
<script>
(function(){{
  var rows=Array.from(document.querySelectorAll('#rows tr'));
  var sel=document.getElementById('cat'), q=document.getElementById('q'),
      count=document.getElementById('count');
  var cats=JSON.parse(document.getElementById('cats').textContent);
  Object.keys(cats).sort().forEach(function(c){{
    var o=document.createElement('option');o.value=c;o.textContent=c;sel.appendChild(o);
  }});
  var sortKey=null, dir=1;
  var isNum={{3:1,4:1,5:1,6:1}};
  function apply(){{
    var f=0;
    rows.forEach(function(tr){{
      var cat=tr.querySelector('.cat').getAttribute('data-cat');
      var hay=tr.textContent.toLowerCase();
      var ok=(sel.value===''||cat===sel.value)&&(q.value===''||hay.indexOf(q.value.toLowerCase())>=0);
      tr.style.display=ok?'':'none'; if(ok)f++;
    }});
    if(sortKey!==null){{
      rows.sort(function(a,b){{
        var av=a.cells[sortKey].textContent.trim(), bv=b.cells[sortKey].textContent.trim();
        if(isNum[sortKey]){{ av=parseFloat(av)||0; bv=parseFloat(bv)||0; }}
        return av<bv?-dir:av>bv?dir:0;
      }});
      rows.forEach(function(tr){{ document.getElementById('rows').appendChild(tr); }});
    }}
    count.textContent=f+' / '+rows.length+' rows';
  }}
  Array.from(document.querySelectorAll('th[data-k]')).forEach(function(th){{
    th.addEventListener('click',function(){{
      var k=parseInt(th.getAttribute('data-k'),10);
      dir=(sortKey===k)?-dir:1; sortKey=k; apply();
    }});
  }});
  sel.addEventListener('change',apply); q.addEventListener('input',apply); apply();
}})();
</script>
</body></html>
"""

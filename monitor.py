"""
╔══════════════════════════════════════════════════════════╗
║  MEMORA :: SYSCTRL MONITOR  //  PORT 8888               ║
║  Cyberpunk system dashboard — local process manager     ║
╚══════════════════════════════════════════════════════════╝

Run:   python monitor.py
Open:  http://localhost:8888
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Set

import psutil
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  —  edit to match your setup
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
API_CMD  = [sys.executable, str(BASE_DIR / "api.py")]
API_CWD  = str(BASE_DIR)
CF_CMD   = ["cloudflared", "tunnel", "--url", "http://localhost:8000"]
CF_CWD   = str(BASE_DIR)
MONITOR_PORT = 8888

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

ANSI_RE   = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
CF_URL_RE = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")

def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)

def detect_level(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("error", "exception", "traceback", "critical", "fail", "err ")):
        return "error"
    if any(w in t for w in ("warning", "warn")):
        return "warn"
    if any(w in t for w in ("started", "ready", "ok", "success", "running",
                             "listening", "connected", "uvicorn", "startup")):
        return "ok"
    return "info"

def get_gpu() -> Optional[dict]:
    """NVIDIA GPU stats via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, creationflags=0x08000000
        )
        if r.returncode == 0 and r.stdout.strip():
            p = [x.strip() for x in r.stdout.strip().split(",")]
            if len(p) >= 5:
                return {
                    "name":      p[0],
                    "load":      float(p[1]),
                    "mem_used":  int(p[2]),
                    "mem_total": int(p[3]),
                    "temp":      int(p[4]),
                }
    except Exception:
        pass
    return None

def proc_stats(proc: Optional[asyncio.subprocess.Process]) -> dict:
    """Get CPU/memory for a running subprocess."""
    if proc is None or proc.returncode is not None:
        return {}
    try:
        ps = psutil.Process(proc.pid)
        return {
            "cpu":    round(ps.cpu_percent(interval=None), 1),
            "mem_mb": round(ps.memory_info().rss / (1024 ** 2), 1),
        }
    except Exception:
        return {}

# ═══════════════════════════════════════════════════════════════════════════════
#  WS MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class WSManager:
    def __init__(self):
        self.clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, data: dict):
        dead = set()
        msg  = json.dumps(data)
        for ws in self.clients:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.clients -= dead

ws = WSManager()

# ═══════════════════════════════════════════════════════════════════════════════
#  PROCESS MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessManager:
    def __init__(self, name: str, cmd: list, cwd: str):
        self.name       = name
        self.cmd        = cmd
        self.cwd        = cwd
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.status     = "stopped"
        self.started_at: Optional[float] = None
        self.cf_url: Optional[str] = None

    @property
    def uptime_s(self) -> Optional[int]:
        return int(time.time() - self.started_at) if self.started_at else None

    async def _log(self, text: str, level: str = "info"):
        await ws.broadcast({"type": "log", "src": "system", "level": level,
                            "text": text, "ts": time.time()})

    async def start(self):
        if self.proc and self.proc.returncode is None:
            await self._log(f"[{self.name}] Already running (pid {self.proc.pid})", "warn")
            return
        await self._log(f"[{self.name}] Starting…", "ok")
        try:
            self.proc = await asyncio.create_subprocess_exec(
                *self.cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.cwd,
            )
            self.status     = "running"
            self.started_at = time.time()
            await ws.broadcast({"type": "proc", "name": self.name,
                                "status": "running", "pid": self.proc.pid})
            asyncio.create_task(self._reader())
        except Exception as exc:
            self.status = "error"
            await self._log(f"[{self.name}] Failed: {exc}", "error")
            await ws.broadcast({"type": "proc", "name": self.name, "status": "error"})

    async def stop(self):
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try:
                await asyncio.wait_for(self.proc.wait(), timeout=6)
            except asyncio.TimeoutError:
                self.proc.kill()
        self.proc       = None
        self.status     = "stopped"
        self.started_at = None
        await ws.broadcast({"type": "proc", "name": self.name, "status": "stopped"})

    async def restart(self):
        await self._log(f"[{self.name}] Restarting…", "warn")
        await self.stop()
        await asyncio.sleep(0.8)
        if self.name == "cloudflared":
            self.cf_url = None
            await ws.broadcast({"type": "cf_url", "url": None})
        await self.start()

    async def _reader(self):
        while self.proc and self.proc.stdout:
            try:
                raw = await self.proc.stdout.readline()
            except Exception:
                break
            if not raw:
                break
            text  = strip_ansi(raw.decode("utf-8", errors="replace")).rstrip()
            if not text:
                continue
            level = detect_level(text)
            await ws.broadcast({"type": "log", "src": self.name,
                                "level": level, "text": text, "ts": time.time()})
            if self.name == "cloudflared":
                m = CF_URL_RE.search(text)
                if m:
                    self.cf_url = m.group()
                    await ws.broadcast({"type": "cf_url", "url": self.cf_url})
        rc           = self.proc.returncode if self.proc else "?"
        self.status  = "stopped"
        self.started_at = None
        await ws.broadcast({"type": "proc", "name": self.name,
                            "status": "stopped", "exit_code": rc})
        await self._log(f"[{self.name}] Exited (code {rc})", "warn")

api_mgr = ProcessManager("api",        API_CMD, API_CWD)
cf_mgr  = ProcessManager("cloudflared", CF_CMD,  CF_CWD)

# ═══════════════════════════════════════════════════════════════════════════════
#  STATS LOOP
# ═══════════════════════════════════════════════════════════════════════════════

_net_prev = (0, 0, time.time())

async def stats_loop():
    global _net_prev
    psutil.cpu_percent(interval=None)  # prime
    while True:
        try:
            cpu  = psutil.cpu_percent(interval=None)
            mem  = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            nio  = psutil.net_io_counters()
            now  = time.time()
            dt   = now - _net_prev[2] or 1
            rx_s = (nio.bytes_recv - _net_prev[0]) / dt / 1024
            tx_s = (nio.bytes_sent - _net_prev[1]) / dt / 1024
            _net_prev = (nio.bytes_recv, nio.bytes_sent, now)
            gpu  = get_gpu()
            papi = proc_stats(api_mgr.proc)
            pcf  = proc_stats(cf_mgr.proc)
            await ws.broadcast({
                "type":       "stats",
                "cpu_pct":    round(cpu, 1),
                "ram_pct":    round(mem.percent, 1),
                "ram_used":   round(mem.used  / (1024**3), 2),
                "ram_total":  round(mem.total / (1024**3), 1),
                "disk_pct":   round(disk.percent, 1),
                "disk_free":  round(disk.free  / (1024**3), 1),
                "rx_kb":      round(rx_s, 1),
                "tx_kb":      round(tx_s, 1),
                "gpu":        gpu,
                "api_proc":   papi,
                "cf_proc":    pcf,
                "api_uptime": api_mgr.uptime_s,
                "cf_uptime":  cf_mgr.uptime_s,
                "ts":         now,
            })
        except Exception:
            pass
        await asyncio.sleep(2)

# ═══════════════════════════════════════════════════════════════════════════════
#  FASTAPI
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Memora Monitor")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(stats_loop())

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await ws.connect(websocket)
    # Send initial state
    await websocket.send_text(json.dumps({
        "type":      "init",
        "api":       {"status": api_mgr.status, "pid": api_mgr.proc.pid if api_mgr.proc else None,
                      "uptime": api_mgr.uptime_s},
        "cf":        {"status": cf_mgr.status,  "url": cf_mgr.cf_url,
                      "uptime": cf_mgr.uptime_s},
    }))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws.disconnect(websocket)

@app.post("/api/start")
async def api_start():   await api_mgr.start();   return {"status": api_mgr.status}

@app.post("/api/stop")
async def api_stop():    await api_mgr.stop();    return {"status": api_mgr.status}

@app.post("/api/restart")
async def api_restart(): await api_mgr.restart(); return {"status": api_mgr.status}

@app.post("/cf/start")
async def cf_start():    await cf_mgr.start();    return {"status": cf_mgr.status}

@app.post("/cf/stop")
async def cf_stop():     await cf_mgr.stop();     return {"status": cf_mgr.status}

@app.post("/cf/restart")
async def cf_restart():  await cf_mgr.restart();  return {"status": cf_mgr.status}

@app.get("/status")
async def get_status():
    return {
        "api": {"status": api_mgr.status,
                "pid":    api_mgr.proc.pid if api_mgr.proc else None,
                "uptime": api_mgr.uptime_s},
        "cf":  {"status": cf_mgr.status,
                "url":    cf_mgr.cf_url,
                "uptime": cf_mgr.uptime_s},
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MEMORA :: SYSCTRL</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #03060d;
  --bg2:      #060c17;
  --bg3:      #0a1525;
  --cyan:     #00d4ff;
  --cyan2:    #00f5c8;
  --magenta:  #ff006e;
  --green:    #39ff14;
  --amber:    #ffb700;
  --red:      #ff2244;
  --purple:   #b44dff;
  --dim:      #2a4060;
  --text:     #7eb8d4;
  --text2:    #4a7a9b;
  --panel:    rgba(6,15,28,0.95);
  --border:   rgba(0,212,255,0.15);
  --glow:     0 0 20px rgba(0,212,255,0.12);
  --font-mono:'Share Tech Mono', monospace;
  --font-hud: 'Orbitron', sans-serif;
}
*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
html,body { height:100%; overflow:hidden; }

body {
  background: var(--bg);
  color: var(--cyan);
  font-family: var(--font-mono);
  font-size: 12px;
}

/* ── Background grid ── */
body::before {
  content:'';
  position:fixed;inset:0;
  background-image:
    linear-gradient(rgba(0,212,255,.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,.04) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events:none; z-index:0;
}
/* Scanlines */
body::after {
  content:'';
  position:fixed;inset:0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 3px,
    rgba(0,0,0,.18) 3px, rgba(0,0,0,.18) 4px
  );
  pointer-events:none; z-index:9999;
}

/* ── Layout ── */
#app {
  position:relative; z-index:1;
  display:flex; flex-direction:column;
  height:100vh; gap:0;
}
#header {
  display:flex; align-items:center; justify-content:space-between;
  padding:0 20px;
  height:48px; flex-shrink:0;
  border-bottom:1px solid var(--border);
  background:rgba(3,6,13,.98);
}
#body {
  flex:1; display:grid;
  grid-template-columns: 320px 1fr;
  grid-template-rows: 1fr;
  gap:1px; overflow:hidden;
  background: var(--dim);
}
#left  { display:flex; flex-direction:column; gap:1px; background:var(--dim); overflow:hidden; }
#right { display:flex; flex-direction:column; overflow:hidden; background:var(--bg2); }

/* ── Panel ── */
.panel {
  background: var(--panel);
  position:relative; overflow:hidden;
  padding:14px 16px;
}
.panel-title {
  font-family:var(--font-hud);
  font-size:9px; letter-spacing:.2em; text-transform:uppercase;
  color:var(--text2); margin-bottom:12px;
  display:flex; align-items:center; gap:8px;
}
.panel-title::after {
  content:''; flex:1; height:1px;
  background:linear-gradient(90deg,var(--border),transparent);
}

/* ── Header pieces ── */
.hud-logo {
  font-family:var(--font-hud);
  font-size:15px; font-weight:900; letter-spacing:.12em;
  background:linear-gradient(135deg,var(--cyan),var(--cyan2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  text-shadow:none;
  filter:drop-shadow(0 0 8px var(--cyan));
}
.hud-logo span { -webkit-text-fill-color: var(--text2); font-weight:400; font-size:11px; }
#hud-time {
  font-family:var(--font-hud); font-size:13px; color:var(--cyan);
  letter-spacing:.1em; min-width:80px; text-align:center;
}
#ws-status {
  display:flex; align-items:center; gap:6px; font-size:10px;
  color:var(--text2); letter-spacing:.1em;
}
#ws-dot {
  width:8px; height:8px; border-radius:50%;
  background:var(--red); transition:background .3s;
}
#ws-dot.on  { background:var(--green); box-shadow:0 0 8px var(--green); animation:pulse 2s infinite; }
#ws-dot.off { background:var(--red);   box-shadow:0 0 8px var(--red); }

/* ── Stat gauges ── */
#stats-grid {
  display:grid; grid-template-columns:repeat(3,1fr); gap:8px;
}
.gauge {
  background:rgba(0,20,40,.6);
  border:1px solid var(--border); border-radius:4px;
  padding:8px 10px; position:relative; overflow:hidden;
}
.gauge::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
  opacity:.4;
}
.g-label { font-size:9px; color:var(--text2); letter-spacing:.15em; text-transform:uppercase; }
.g-val   { font-family:var(--font-hud); font-size:20px; font-weight:700;
           margin:2px 0; line-height:1; transition:color .4s; }
.g-sub   { font-size:9px; color:var(--text2); }
.g-bar   { margin-top:6px; height:3px; background:rgba(255,255,255,.06); border-radius:2px; overflow:hidden; }
.g-fill  { height:100%; border-radius:2px; transition:width .6s cubic-bezier(.4,0,.2,1); }

/* Sparkline canvas */
.g-spark { margin-top:4px; width:100%; height:28px; display:block; }

.g-cpu   { --gc:var(--cyan);    } .g-cpu   .g-val,.g-cpu   .g-fill { color:var(--cyan);    background:var(--cyan); }
.g-ram   { --gc:var(--cyan2);   } .g-ram   .g-val,.g-ram   .g-fill { color:var(--cyan2);   background:var(--cyan2); }
.g-disk  { --gc:var(--purple);  } .g-disk  .g-val,.g-disk  .g-fill { color:var(--purple);  background:var(--purple); }
.g-gpu   { --gc:var(--green);   } .g-gpu   .g-val,.g-gpu   .g-fill { color:var(--green);   background:var(--green); }
.g-vram  { --gc:var(--amber);   } .g-vram  .g-val,.g-vram  .g-fill { color:var(--amber);   background:var(--amber); }
.g-temp  { --gc:var(--magenta); } .g-temp  .g-val,.g-temp  .g-fill { color:var(--magenta); background:var(--magenta); }

/* ── Process cards ── */
.proc-card {
  background:rgba(0,20,40,.5);
  border:1px solid var(--border); border-radius:4px; padding:12px;
  display:flex; flex-direction:column; gap:8px;
}
.proc-header { display:flex; align-items:center; gap:8px; }
.proc-dot {
  width:10px; height:10px; border-radius:50%; flex-shrink:0;
  transition:background .3s, box-shadow .3s;
}
.proc-dot.running { background:var(--green); box-shadow:0 0 10px var(--green); animation:pulse 2s infinite; }
.proc-dot.stopped { background:var(--dim); }
.proc-dot.error   { background:var(--red); box-shadow:0 0 10px var(--red); }
.proc-name { font-family:var(--font-hud); font-size:11px; letter-spacing:.08em; flex:1; }
.proc-pid  { font-size:9px; color:var(--text2); }
.proc-meta { font-size:9px; color:var(--text2); display:flex; gap:12px; }
.proc-btns { display:flex; gap:6px; }

/* ── Buttons ── */
.btn {
  font-family:var(--font-hud); font-size:9px; letter-spacing:.12em;
  padding:5px 12px; border-radius:3px; border:none; cursor:pointer;
  text-transform:uppercase; transition:all .2s; position:relative; overflow:hidden;
}
.btn::after {
  content:''; position:absolute; inset:0;
  background:rgba(255,255,255,.08);
  opacity:0; transition:opacity .15s;
}
.btn:hover::after { opacity:1; }
.btn-start  { background:rgba(57,255,20,.12);  border:1px solid rgba(57,255,20,.4);  color:var(--green);   }
.btn-stop   { background:rgba(255,34,68,.12);  border:1px solid rgba(255,34,68,.4);  color:var(--red);     }
.btn-restart{ background:rgba(255,183,0,.12);  border:1px solid rgba(255,183,0,.4);  color:var(--amber);   }

/* ── CF URL box ── */
#cf-url-box {
  margin-top:4px;
  background:rgba(0,212,255,.06);
  border:1px solid rgba(0,212,255,.25);
  border-radius:4px; padding:10px 12px;
  position:relative; overflow:hidden;
}
#cf-url-box::before {
  content:''; position:absolute; top:0;left:0;right:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
}
.cf-label { font-size:9px; color:var(--text2); letter-spacing:.15em; text-transform:uppercase; margin-bottom:4px; }
#cf-url-text {
  font-size:11px; color:var(--cyan);
  word-break:break-all; line-height:1.5;
  text-shadow: 0 0 8px var(--cyan);
  cursor:pointer;
}
#cf-url-text:hover { text-decoration:underline; }
.cf-copy-hint { font-size:9px; color:var(--text2); margin-top:4px; }

/* ── Net stats ── */
.net-row { display:flex; gap:16px; }
.net-item { flex:1; background:rgba(0,20,40,.5); border:1px solid var(--border);
            border-radius:4px; padding:8px 10px; }
.net-label { font-size:9px; color:var(--text2); letter-spacing:.1em; }
.net-val   { font-family:var(--font-hud); font-size:14px; color:var(--cyan); margin-top:2px; }

/* ── Log panel ── */
#log-header {
  padding:10px 14px;
  display:flex; align-items:center; gap:10px; flex-shrink:0;
  border-bottom:1px solid var(--border);
  background:rgba(3,6,13,.98);
}
#log-title { font-family:var(--font-hud); font-size:10px; letter-spacing:.15em; color:var(--text2); }
.log-filter {
  font-family:var(--font-hud); font-size:8px; letter-spacing:.12em;
  padding:3px 10px; border-radius:2px; border:none; cursor:pointer;
  text-transform:uppercase; background:transparent;
  border:1px solid var(--border); color:var(--text2); transition:all .2s;
}
.log-filter.active { background:rgba(0,212,255,.12); border-color:var(--cyan); color:var(--cyan); }
.log-filter:hover  { border-color:rgba(0,212,255,.4); }
#log-count { margin-left:auto; font-size:10px; color:var(--text2); }
.log-clear { background:transparent; border:1px solid rgba(255,34,68,.3); color:var(--red);
             font-family:var(--font-hud); font-size:8px; letter-spacing:.1em; padding:3px 8px;
             border-radius:2px; cursor:pointer; text-transform:uppercase; }
.log-clear:hover { background:rgba(255,34,68,.12); }

#log-stream {
  flex:1; overflow-y:auto; padding:6px 14px;
  font-size:11px; line-height:1.65;
  background:var(--bg); scroll-behavior:smooth;
}
#log-stream::-webkit-scrollbar { width:4px; }
#log-stream::-webkit-scrollbar-track { background:transparent; }
#log-stream::-webkit-scrollbar-thumb { background:var(--dim); border-radius:2px; }

.log-line {
  display:flex; align-items:flex-start; gap:8px;
  padding:1px 0; border-bottom:1px solid rgba(255,255,255,.02);
  animation:fadeIn .2s ease;
}
@keyframes fadeIn { from{opacity:0;transform:translateX(-4px)} to{opacity:1;transform:none} }

.log-ts   { color:var(--text2); flex-shrink:0; font-size:10px; }
.log-src  { flex-shrink:0; font-size:10px; min-width:80px; text-align:right;
            padding:0 4px; border-radius:2px; }
.log-txt  { flex:1; word-break:break-word; }

.log-src.api         { color:#00d4ff; background:rgba(0,212,255,.08); }
.log-src.cloudflared { color:#b44dff; background:rgba(180,77,255,.08); }
.log-src.system      { color:#ffb700; background:rgba(255,183,0,.08); }

.level-info  .log-txt { color:#5a8faa; }
.level-ok    .log-txt { color:var(--green); }
.level-warn  .log-txt { color:var(--amber); }
.level-error .log-txt { color:var(--red); text-shadow:0 0 6px rgba(255,34,68,.4); }

/* ── Animations ── */
@keyframes pulse {
  0%,100%{opacity:1} 50%{opacity:.5}
}
@keyframes glow-cycle {
  0%,100%{box-shadow:0 0 10px rgba(0,212,255,.2)}
  50%    {box-shadow:0 0 24px rgba(0,212,255,.5)}
}
.glow-anim { animation:glow-cycle 3s ease-in-out infinite; }

/* ── Misc ── */
.tag {
  display:inline-block; font-size:8px; letter-spacing:.1em;
  padding:1px 6px; border-radius:2px; text-transform:uppercase;
  font-family:var(--font-hud);
}
.tag-run  { background:rgba(57,255,20,.12);  color:var(--green);   border:1px solid rgba(57,255,20,.3); }
.tag-stop { background:rgba(42,64,96,.4);    color:var(--text2);   border:1px solid var(--border); }
.tag-err  { background:rgba(255,34,68,.12);  color:var(--red);     border:1px solid rgba(255,34,68,.3); }

.hbar { width:100%; height:1px; background:var(--border); margin:10px 0; }

#copied-toast {
  position:fixed; bottom:20px; right:20px; z-index:9998;
  background:rgba(0,212,255,.15); border:1px solid var(--cyan);
  color:var(--cyan); font-family:var(--font-hud); font-size:11px; letter-spacing:.1em;
  padding:8px 18px; border-radius:4px;
  opacity:0; transform:translateY(6px); pointer-events:none;
  transition:all .3s;
}
#copied-toast.show { opacity:1; transform:none; }
</style>
</head>
<body>
<div id="app">

  <!-- ─── HEADER ─── -->
  <div id="header">
    <div class="hud-logo">MEMORA <span>:: SYSCTRL // v2.0</span></div>
    <div id="hud-time">00:00:00</div>
    <div id="ws-status">
      <div id="ws-dot" class="off"></div>
      <span id="ws-label">OFFLINE</span>
    </div>
  </div>

  <!-- ─── BODY ─── -->
  <div id="body">

    <!-- ═══ LEFT COLUMN ═══ -->
    <div id="left">

      <!-- SYSTEM VITALS -->
      <div class="panel" style="flex-shrink:0">
        <div class="panel-title">▸ system vitals</div>
        <div id="stats-grid">
          <div class="gauge g-cpu">
            <div class="g-label">CPU</div>
            <div class="g-val" id="cpu-val">--</div>
            <div class="g-sub" id="cpu-sub">%</div>
            <div class="g-bar"><div class="g-fill" id="cpu-bar" style="width:0%"></div></div>
            <canvas class="g-spark" id="spark-cpu" width="90" height="28"></canvas>
          </div>
          <div class="gauge g-ram">
            <div class="g-label">RAM</div>
            <div class="g-val" id="ram-val">--</div>
            <div class="g-sub" id="ram-sub">%</div>
            <div class="g-bar"><div class="g-fill" id="ram-bar" style="width:0%"></div></div>
            <canvas class="g-spark" id="spark-ram" width="90" height="28"></canvas>
          </div>
          <div class="gauge g-disk">
            <div class="g-label">DISK</div>
            <div class="g-val" id="disk-val">--</div>
            <div class="g-sub" id="disk-sub">%</div>
            <div class="g-bar"><div class="g-fill" id="disk-bar" style="width:0%"></div></div>
            <canvas class="g-spark" id="spark-disk" width="90" height="28"></canvas>
          </div>
          <div class="gauge g-gpu">
            <div class="g-label">GPU</div>
            <div class="g-val" id="gpu-val">--</div>
            <div class="g-sub">%</div>
            <div class="g-bar"><div class="g-fill" id="gpu-bar" style="width:0%"></div></div>
            <canvas class="g-spark" id="spark-gpu" width="90" height="28"></canvas>
          </div>
          <div class="gauge g-vram">
            <div class="g-label">VRAM</div>
            <div class="g-val" id="vram-val">--</div>
            <div class="g-sub" id="vram-sub">MB</div>
            <div class="g-bar"><div class="g-fill" id="vram-bar" style="width:0%"></div></div>
            <canvas class="g-spark" id="spark-vram" width="90" height="28"></canvas>
          </div>
          <div class="gauge g-temp">
            <div class="g-label">GPU TEMP</div>
            <div class="g-val" id="temp-val">--</div>
            <div class="g-sub">°C</div>
            <div class="g-bar"><div class="g-fill" id="temp-bar" style="width:0%"></div></div>
            <canvas class="g-spark" id="spark-temp" width="90" height="28"></canvas>
          </div>
        </div>
      </div>

      <!-- NETWORK -->
      <div class="panel" style="flex-shrink:0">
        <div class="panel-title">▸ network i/o</div>
        <div class="net-row">
          <div class="net-item">
            <div class="net-label">▼ RECV</div>
            <div class="net-val" id="net-rx">-- KB/s</div>
          </div>
          <div class="net-item">
            <div class="net-label">▲ SEND</div>
            <div class="net-val" id="net-tx">-- KB/s</div>
          </div>
        </div>
      </div>

      <!-- PROCESS MATRIX -->
      <div class="panel" style="flex:1;overflow-y:auto">
        <div class="panel-title">▸ process matrix</div>

        <!-- api.py -->
        <div class="proc-card" id="card-api">
          <div class="proc-header">
            <div class="proc-dot" id="dot-api"></div>
            <div class="proc-name">api.py</div>
            <span class="tag tag-stop" id="tag-api">STOPPED</span>
            <div class="proc-pid" id="pid-api"></div>
          </div>
          <div class="proc-meta" id="meta-api">
            <span id="up-api">uptime: --</span>
            <span id="cpu-api">cpu: --</span>
            <span id="mem-api">mem: --</span>
          </div>
          <div class="proc-btns">
            <button class="btn btn-start"   onclick="cmd('/api/start')">▶ Start</button>
            <button class="btn btn-stop"    onclick="cmd('/api/stop')">■ Stop</button>
            <button class="btn btn-restart" onclick="cmd('/api/restart')">↺ Restart</button>
          </div>
        </div>

        <div style="height:10px"></div>

        <!-- cloudflared -->
        <div class="proc-card" id="card-cf">
          <div class="proc-header">
            <div class="proc-dot" id="dot-cf"></div>
            <div class="proc-name">cloudflared</div>
            <span class="tag tag-stop" id="tag-cf">STOPPED</span>
            <div class="proc-pid" id="pid-cf"></div>
          </div>
          <div class="proc-meta" id="meta-cf">
            <span id="up-cf">uptime: --</span>
            <span id="cpu-cf">cpu: --</span>
            <span id="mem-cf">mem: --</span>
          </div>
          <div class="proc-btns">
            <button class="btn btn-start"   onclick="cmd('/cf/start')">▶ Start</button>
            <button class="btn btn-stop"    onclick="cmd('/cf/stop')">■ Stop</button>
            <button class="btn btn-restart" onclick="cmd('/cf/restart')">↺ Restart</button>
          </div>
        </div>

        <!-- Cloudflare URL -->
        <div style="height:12px"></div>
        <div id="cf-url-box">
          <div class="cf-label">tunnel url</div>
          <div id="cf-url-text" onclick="copyCfUrl()" title="Click to copy">
            waiting for cloudflared…
          </div>
          <div class="cf-copy-hint">click to copy</div>
        </div>

        <!-- GPU name -->
        <div style="height:10px"></div>
        <div style="font-size:9px;color:var(--text2);letter-spacing:.1em">
          GPU: <span id="gpu-name" style="color:var(--text)">detecting…</span>
        </div>
      </div>

    </div><!-- /left -->

    <!-- ═══ RIGHT COLUMN — LOGS ═══ -->
    <div id="right">
      <div id="log-header">
        <span id="log-title">LOG STREAM</span>
        <button class="log-filter active" data-src="all"         onclick="setFilter('all',this)">ALL</button>
        <button class="log-filter"        data-src="api"         onclick="setFilter('api',this)">API</button>
        <button class="log-filter"        data-src="cloudflared" onclick="setFilter('cloudflared',this)">TUNNEL</button>
        <button class="log-filter"        data-src="system"      onclick="setFilter('system',this)">SYS</button>
        <span id="log-count" style="font-family:var(--font-hud);font-size:10px;color:var(--text2)">0 lines</span>
        <label style="display:flex;align-items:center;gap:4px;font-size:9px;color:var(--text2);cursor:pointer;margin-left:4px">
          <input type="checkbox" id="auto-scroll" checked style="accent-color:var(--cyan)"> FOLLOW
        </label>
        <button class="log-clear" onclick="clearLogs()">CLEAR</button>
      </div>
      <div id="log-stream"></div>
    </div>

  </div><!-- /body -->
</div><!-- /app -->

<div id="copied-toast">URL COPIED</div>

<script>
// ─── State ───
let logFilter   = 'all';
let logLines    = [];
const MAX_LINES = 800;
const sparks    = { cpu:[], ram:[], disk:[], gpu:[], vram:[], temp:[] };
const MAX_SPARK = 60;
let cfUrl       = null;

// ─── Clock ───
function tick() {
  const now = new Date();
  document.getElementById('hud-time').textContent =
    now.toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
setInterval(tick, 1000); tick();

// ─── WebSocket ───
let ws, reconnectTimer;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('ws-dot').className   = 'on';
    document.getElementById('ws-label').textContent = 'ONLINE';
    sysLog('WebSocket connected', 'ok');
    clearTimeout(reconnectTimer);
  };

  ws.onclose = () => {
    document.getElementById('ws-dot').className   = 'off';
    document.getElementById('ws-label').textContent = 'OFFLINE';
    sysLog('WebSocket disconnected — reconnecting…', 'warn');
    reconnectTimer = setTimeout(connect, 3000);
  };

  ws.onerror = () => ws.close();

  ws.onmessage = e => {
    try { handle(JSON.parse(e.data)); } catch(_) {}
  };
}
connect();

// ─── Message handler ───
function handle(msg) {
  switch(msg.type) {
    case 'init':
      updateProc('api', msg.api);
      updateProc('cf',  msg.cf);
      if (msg.cf?.url) setUrl(msg.cf.url);
      break;

    case 'stats':
      updateStats(msg); break;

    case 'proc':
      const key = msg.name === 'cloudflared' ? 'cf' : msg.name;
      updateProc(key, msg);
      break;

    case 'log':
      addLog(msg); break;

    case 'cf_url':
      if (msg.url) setUrl(msg.url); else clearUrl(); break;
  }
}

// ─── Stats ───
function setGauge(id, val, pct, sub) {
  const el = document.getElementById(id+'-val');
  if (el) el.textContent = val;
  const bar = document.getElementById(id+'-bar');
  if (bar) bar.style.width = Math.min(pct,100)+'%';
  if (sub !== undefined) {
    const s = document.getElementById(id+'-sub');
    if (s) s.textContent = sub;
  }
}

function updateStats(s) {
  setGauge('cpu',  s.cpu_pct+'%',  s.cpu_pct,  s.cpu_pct+'%');
  setGauge('ram',  s.ram_pct+'%',  s.ram_pct,  s.ram_used+'G / '+s.ram_total+'G');
  setGauge('disk', s.disk_pct+'%', s.disk_pct, s.disk_free+'G free');

  if (s.gpu) {
    setGauge('gpu',  s.gpu.load+'%',     s.gpu.load,
             s.gpu.load+'%');
    setGauge('vram', s.gpu.mem_used+'M', (s.gpu.mem_used/s.gpu.mem_total)*100,
             s.gpu.mem_used+' / '+s.gpu.mem_total+' MB');
    const tp = Math.min((s.gpu.temp/90)*100, 100);
    setGauge('temp', s.gpu.temp+'°', tp, s.gpu.temp+'°C');
    document.getElementById('gpu-name').textContent = s.gpu.name;
  } else {
    ['gpu','vram','temp'].forEach(k => {
      const v = document.getElementById(k+'-val');
      if (v && v.textContent==='--') v.textContent = 'N/A';
    });
    document.getElementById('gpu-name').textContent = 'not detected';
  }

  document.getElementById('net-rx').textContent = s.rx_kb + ' KB/s';
  document.getElementById('net-tx').textContent = s.tx_kb + ' KB/s';

  // Proc-level stats
  if (s.api_proc && Object.keys(s.api_proc).length) {
    document.getElementById('cpu-api').textContent = 'cpu: '+s.api_proc.cpu+'%';
    document.getElementById('mem-api').textContent = 'mem: '+s.api_proc.mem_mb+' MB';
  }
  if (s.cf_proc && Object.keys(s.cf_proc).length) {
    document.getElementById('cpu-cf').textContent = 'cpu: '+s.cf_proc.cpu+'%';
    document.getElementById('mem-cf').textContent = 'mem: '+s.cf_proc.mem_mb+' MB';
  }
  if (s.api_uptime != null) document.getElementById('up-api').textContent = 'uptime: '+fmtUp(s.api_uptime);
  if (s.cf_uptime  != null) document.getElementById('up-cf').textContent  = 'uptime: '+fmtUp(s.cf_uptime);

  // Sparklines
  pushSpark('cpu',  s.cpu_pct);
  pushSpark('ram',  s.ram_pct);
  pushSpark('disk', s.disk_pct);
  pushSpark('gpu',  s.gpu ? s.gpu.load  : 0);
  pushSpark('vram', s.gpu ? (s.gpu.mem_used/s.gpu.mem_total)*100 : 0);
  pushSpark('temp', s.gpu ? Math.min((s.gpu.temp/90)*100,100) : 0);
}

function fmtUp(s) {
  if (s < 60) return s+'s';
  if (s < 3600) return Math.floor(s/60)+'m '+s%60+'s';
  return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';
}

// ─── Sparklines ───
const SPARK_COLORS = {
  cpu:'#00d4ff', ram:'#00f5c8', disk:'#b44dff',
  gpu:'#39ff14', vram:'#ffb700', temp:'#ff006e'
};

function pushSpark(key, val) {
  const arr = sparks[key];
  arr.push(val);
  if (arr.length > MAX_SPARK) arr.shift();
  drawSpark(key, arr);
}

function drawSpark(key, arr) {
  const c = document.getElementById('spark-'+key);
  if (!c) return;
  const ctx = c.getContext('2d');
  const W = c.clientWidth || c.width;
  const H = c.clientHeight || c.height;
  c.width = W; c.height = H;
  ctx.clearRect(0,0,W,H);
  if (arr.length < 2) return;

  const max = 100, step = W / (arr.length - 1);
  ctx.beginPath();
  arr.forEach((v,i) => {
    const x = i * step;
    const y = H - (v / max) * H;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = SPARK_COLORS[key];
  ctx.lineWidth   = 1.5;
  ctx.stroke();

  // Fill
  ctx.lineTo((arr.length-1)*step, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0, SPARK_COLORS[key]+'55');
  grad.addColorStop(1, 'transparent');
  ctx.fillStyle = grad;
  ctx.fill();
}

// ─── Process status ───
function updateProc(key, data) {
  if (!data) return;
  const st   = data.status;
  const dot  = document.getElementById('dot-'+key);
  const tag  = document.getElementById('tag-'+key);
  const pid  = document.getElementById('pid-'+key);
  if (!dot) return;

  dot.className = 'proc-dot ' + (st || 'stopped');
  if (tag) {
    tag.textContent  = (st || 'STOPPED').toUpperCase();
    tag.className    = 'tag ' + (st==='running'?'tag-run':st==='error'?'tag-err':'tag-stop');
  }
  if (pid) pid.textContent = data.pid ? 'PID '+data.pid : '';
}

// ─── CF URL ───
function setUrl(url) {
  cfUrl = url;
  const el = document.getElementById('cf-url-text');
  el.textContent = url;
  el.style.color = 'var(--cyan)';
}
function clearUrl() {
  cfUrl = null;
  document.getElementById('cf-url-text').textContent = 'waiting for cloudflared…';
}
function copyCfUrl() {
  if (!cfUrl) return;
  navigator.clipboard.writeText(cfUrl).then(() => {
    const t = document.getElementById('copied-toast');
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 1800);
  });
}

// ─── Logs ───
function sysLog(text, level='info') {
  addLog({ src:'system', level, text, ts: Date.now()/1000 });
}

function fmtTs(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
}

function addLog(msg) {
  logLines.push(msg);
  if (logLines.length > MAX_LINES + 50) logLines = logLines.slice(-MAX_LINES);
  if (logFilter === 'all' || logFilter === msg.src) renderLine(msg);
  document.getElementById('log-count').textContent = logLines.length + ' lines';
}

function renderLine(msg) {
  const stream = document.getElementById('log-stream');
  const div    = document.createElement('div');
  div.className = 'log-line level-' + (msg.level||'info');
  div.innerHTML =
    `<span class="log-ts">${fmtTs(msg.ts)}</span>` +
    `<span class="log-src ${msg.src}">[${(msg.src||'sys').toUpperCase()}]</span>` +
    `<span class="log-txt">${escHtml(msg.text)}</span>`;
  stream.appendChild(div);
  const follow = document.getElementById('auto-scroll');
  if (follow && follow.checked) stream.scrollTop = stream.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setFilter(src, btn) {
  logFilter = src;
  document.querySelectorAll('.log-filter').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const stream = document.getElementById('log-stream');
  stream.innerHTML = '';
  const lines = src === 'all' ? logLines : logLines.filter(l => l.src === src);
  lines.forEach(renderLine);
}

function clearLogs() {
  logLines = [];
  document.getElementById('log-stream').innerHTML = '';
  document.getElementById('log-count').textContent = '0 lines';
}

// ─── Process commands ───
async function cmd(path) {
  try {
    const r = await fetch(path, {method:'POST'});
    const d = await r.json();
    sysLog('→ '+path+' → '+JSON.stringify(d), 'ok');
  } catch(e) {
    sysLog('Command failed: '+e, 'error');
  }
}
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import socket
    host = socket.gethostname()
    ip   = socket.gethostbyname(host)
    print("\n" + "═"*55)
    print("  MEMORA :: SYSCTRL MONITOR")
    print("═"*55)
    print(f"  Local:   http://localhost:{MONITOR_PORT}")
    print(f"  Network: http://{ip}:{MONITOR_PORT}")
    print(f"  API dir: {BASE_DIR}")
    print("═"*55 + "\n")
    uvicorn.run("monitor:app", host="0.0.0.0", port=MONITOR_PORT,
                reload=False, log_level="warning")

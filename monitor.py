"""
MEMORA :: SYSCTRL MONITOR  //  PORT 8888
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
API_CMD  = [sys.executable, "-u", str(BASE_DIR / "api.py")]  # -u = unbuffered stdout
API_CWD  = str(BASE_DIR)
CF_CWD   = str(BASE_DIR)
MONITOR_PORT = 8888

def _find_cloudflared() -> list:
    """Locate cloudflared binary — checks PATH first, then common Windows locations."""
    import shutil
    CF_ARGS = [
        "tunnel", "--url", "http://localhost:8000",
        "--proxy-connect-timeout", "600s",
        "--proxy-keepalive-timeout", "600s",
    ]
    found = shutil.which("cloudflared")
    if found:
        return [found] + CF_ARGS
    if os.name == "nt":
        candidates = [
            r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
            r"C:\Program Files\cloudflared\cloudflared.exe",
            str(BASE_DIR / "cloudflared.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\cloudflared\cloudflared.exe"),
            os.path.expandvars(r"%USERPROFILE%\cloudflared.exe"),
            r"C:\Windows\cloudflared.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                return [c] + CF_ARGS
    return ["cloudflared"] + CF_ARGS

CF_CMD = _find_cloudflared()

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

# Ring buffer — last 500 log lines sent to new clients on connect
_LOG_HISTORY: list = []
_LOG_HISTORY_MAX = 500

def _store_log(entry: dict):
    _LOG_HISTORY.append(entry)
    if len(_LOG_HISTORY) > _LOG_HISTORY_MAX + 50:
        del _LOG_HISTORY[:-_LOG_HISTORY_MAX]

async def _kill_port(port: int):
    """Terminate any process listening on the given TCP port."""
    try:
        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr and conn.laddr.port == port and conn.status in ("LISTEN", "ESTABLISHED"):
                try:
                    psutil.Process(conn.pid).terminate()
                    await asyncio.sleep(0.6)
                except Exception:
                    pass
    except Exception:
        pass

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
        entry = {"type": "log", "src": "system", "level": level,
                 "text": text, "ts": time.time()}
        _store_log(entry)
        await ws.broadcast(entry)

    async def start(self):
        if self.proc and self.proc.returncode is None:
            await self._log(f"[{self.name}] Already running (pid {self.proc.pid})", "warn")
            return
        # Kill any stray process on port 8000 before launching api
        if self.name == "api":
            await self._log("[api] Clearing port 8000...", "info")
            await _kill_port(8000)
        await self._log(f"[{self.name}] Starting...", "ok")
        try:
            kwargs: dict = dict(
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self.cwd,
                env=os.environ.copy(),  # inherit PATH so cloudflared/python are found
            )
            # On Windows: don't pop open a new console window
            if os.name == "nt":
                kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
            self.proc = await asyncio.create_subprocess_exec(*self.cmd, **kwargs)
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
        await self._log(f"[{self.name}] Restarting...", "warn")
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
            entry = {"type": "log", "src": self.name,
                     "level": level, "text": text, "ts": time.time()}
            _store_log(entry)
            await ws.broadcast(entry)
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

_net_prev  = (0, 0, time.time())
_DISK_PATH = str(Path(BASE_DIR).anchor)  # "C:\" on Windows, "/" on Linux

async def stats_loop():
    global _net_prev
    psutil.cpu_percent(interval=None)  # prime
    while True:
        try:
            cpu  = psutil.cpu_percent(interval=None)
            mem  = psutil.virtual_memory()
            disk = psutil.disk_usage(_DISK_PATH)
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
    # Send current process state
    await websocket.send_text(json.dumps({
        "type": "init",
        "api":  {"status": api_mgr.status, "pid": api_mgr.proc.pid if api_mgr.proc else None,
                 "uptime": api_mgr.uptime_s},
        "cf":   {"status": cf_mgr.status,  "url": cf_mgr.cf_url,
                 "uptime": cf_mgr.uptime_s},
    }))
    # Replay log history so client sees past lines immediately
    for entry in _LOG_HISTORY[-_LOG_HISTORY_MAX:]:
        try:
            await websocket.send_text(json.dumps(entry))
        except Exception:
            break
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
        "api": {"status": api_mgr.status, "pid": api_mgr.proc.pid if api_mgr.proc else None,
                "uptime": api_mgr.uptime_s},
        "cf":  {"status": cf_mgr.status, "url": cf_mgr.cf_url, "uptime": cf_mgr.uptime_s},
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)

# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD HTML  —  readable from 1.5 m, cyberpunk aesthetic
# ═══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MEMORA :: SYSCTRL</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600;900&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #04080f;
  --bg2:       #07101c;
  --bg3:       #0b1828;
  --panel:     #060e1a;
  --border:    rgba(0,200,255,.22);
  --cyan:      #00c8ff;
  --cyan-dim:  rgba(0,200,255,.12);
  --green:     #00e676;
  --green-dim: rgba(0,230,118,.12);
  --amber:     #ffb300;
  --amber-dim: rgba(255,179,0,.12);
  --red:       #ff5252;
  --red-dim:   rgba(255,82,82,.12);
  --purple:    #c060ff;
  --text:      #cce4f5;
  --text2:     #6a9ab5;
  --text3:     #3d6a88;
  --mono:      'JetBrains Mono', monospace;
  --hud:       'Orbitron', sans-serif;
}

*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
html,body { height:100%; overflow:hidden; background:var(--bg); color:var(--text); font-family:var(--mono); }

body::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(0,200,255,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,200,255,.03) 1px, transparent 1px);
  background-size:60px 60px;
}

/* ── App shell: column flex ── */
#app { position:relative; z-index:1; display:flex; flex-direction:column; height:100vh; }

/* ── Header ── */
#hdr {
  display:flex; align-items:center; gap:20px;
  padding:0 20px; height:50px; flex-shrink:0;
  background:var(--bg2); border-bottom:2px solid var(--border);
}
.logo { font-family:var(--hud); font-size:17px; font-weight:900; letter-spacing:.14em; color:var(--cyan); text-shadow:0 0 18px rgba(0,200,255,.5); }
.logo span { font-weight:600; font-size:12px; color:var(--text2); letter-spacing:.08em; }
#hdr-clock { font-family:var(--hud); font-size:17px; font-weight:600; color:var(--text); letter-spacing:.12em; }
.ws-pill {
  display:flex; align-items:center; gap:7px; padding:4px 14px; border-radius:20px;
  border:1px solid var(--border); background:var(--cyan-dim);
  font-family:var(--hud); font-size:11px; letter-spacing:.1em; color:var(--text2);
}
#ws-dot { width:9px; height:9px; border-radius:50%; background:var(--red); transition:all .3s; }
#ws-dot.on { background:var(--green); box-shadow:0 0 10px var(--green); animation:blink 2s infinite; }

/* ── Vitals strip: full width across top ── */
#vitals-strip {
  flex-shrink:0; background:var(--panel);
  padding:10px 16px 12px;
  border-bottom:2px solid var(--border);
}
#stat-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:8px; }

.gauge {
  background:var(--bg3); border:1px solid var(--border); border-radius:6px;
  padding:10px 14px; display:flex; flex-direction:column;
}
.g-lbl { font-family:var(--hud); font-size:10px; font-weight:600; letter-spacing:.18em; color:var(--text3); margin-bottom:2px; }
.g-num { font-family:var(--hud); font-size:32px; font-weight:900; line-height:1.05; letter-spacing:-.01em; transition:color .4s; }
.g-sub { font-size:11px; color:var(--text2); margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.g-bar  { margin-top:6px; height:4px; background:rgba(255,255,255,.07); border-radius:3px; overflow:hidden; }
.g-fill { height:100%; border-radius:3px; transition:width .7s cubic-bezier(.4,0,.2,1); }
.g-spark{ display:block; width:100%; height:22px; margin-top:5px; }

.gc .g-num,.gc .g-fill { color:var(--cyan);   background:var(--cyan); }
.gr .g-num,.gr .g-fill { color:#00e5c8;        background:#00e5c8; }
.gd .g-num,.gd .g-fill { color:var(--purple);  background:var(--purple); }
.gg .g-num,.gg .g-fill { color:var(--green);   background:var(--green); }
.gv .g-num,.gv .g-fill { color:var(--amber);   background:var(--amber); }
.gt .g-num,.gt .g-fill { color:var(--red);     background:var(--red); }

/* ── Bottom area: left panel + logs ── */
#bottom {
  flex:1; display:grid; grid-template-columns:400px 1fr;
  gap:2px; overflow:hidden; background:rgba(0,200,255,.07);
}

#left {
  overflow-y:auto; background:var(--bg);
  display:flex; flex-direction:column; gap:2px;
}
#left::-webkit-scrollbar { width:4px; }
#left::-webkit-scrollbar-track { background:transparent; }
#left::-webkit-scrollbar-thumb { background:var(--bg3); border-radius:2px; }

#right { display:flex; flex-direction:column; overflow:hidden; background:var(--bg); }

/* ── Panel ── */
.panel { background:var(--panel); padding:14px 16px; flex-shrink:0; }
.ptitle {
  font-family:var(--hud); font-size:11px; font-weight:600; letter-spacing:.2em;
  text-transform:uppercase; color:var(--text3); margin-bottom:12px;
  display:flex; align-items:center; gap:10px;
}
.ptitle::after { content:''; flex:1; height:1px; background:var(--border); }

/* ── Network ── */
.net-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
.net-card { background:var(--bg3); border:1px solid var(--border); border-radius:6px; padding:10px 12px; }
.net-lbl { font-family:var(--hud); font-size:10px; letter-spacing:.15em; color:var(--text3); margin-bottom:3px; }
.net-val { font-family:var(--hud); font-size:20px; font-weight:600; color:var(--text); }

/* ── Process cards ── */
.proc-card { background:var(--bg3); border:1px solid var(--border); border-radius:6px; padding:14px; }
.proc-card + .proc-card { margin-top:8px; }
.proc-top { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.proc-dot { width:12px; height:12px; border-radius:50%; flex-shrink:0; transition:all .3s; }
.proc-dot.running { background:var(--green); box-shadow:0 0 10px var(--green); animation:blink 2s infinite; }
.proc-dot.stopped { background:var(--text3); }
.proc-dot.error   { background:var(--red); box-shadow:0 0 10px var(--red); }
.proc-name { font-family:var(--hud); font-size:14px; font-weight:600; color:var(--text); flex:1; }
.proc-badge { font-family:var(--hud); font-size:10px; letter-spacing:.1em; padding:3px 10px; border-radius:3px; }
.pb-run  { background:var(--green-dim); color:var(--green); border:1px solid rgba(0,230,118,.35); }
.pb-stop { background:rgba(255,255,255,.04); color:var(--text3); border:1px solid rgba(255,255,255,.1); }
.pb-err  { background:var(--red-dim); color:var(--red); border:1px solid rgba(255,82,82,.35); }
.proc-meta { display:flex; gap:12px; flex-wrap:wrap; font-size:12px; color:var(--text2); margin-bottom:10px; }
.proc-btns { display:flex; gap:7px; }

.btn {
  font-family:var(--hud); font-size:11px; font-weight:600; letter-spacing:.1em;
  padding:7px 16px; border-radius:4px; border:none; cursor:pointer;
  text-transform:uppercase; transition:all .15s;
}
.btn:hover { filter:brightness(1.2); transform:translateY(-1px); }
.btn:active { transform:translateY(0); filter:brightness(.9); }
.btn-go  { background:var(--green-dim); border:1px solid rgba(0,230,118,.5);  color:var(--green); }
.btn-stp { background:var(--red-dim);   border:1px solid rgba(255,82,82,.5);   color:var(--red); }
.btn-rst { background:var(--amber-dim); border:1px solid rgba(255,179,0,.5);   color:var(--amber); }

/* ── CF URL ── */
#cf-box {
  margin-top:10px; background:rgba(0,200,255,.07);
  border:1px solid rgba(0,200,255,.3); border-radius:6px; padding:12px 14px;
}
.cf-lbl { font-family:var(--hud); font-size:10px; letter-spacing:.18em; color:var(--text3); margin-bottom:5px; }
#cf-url { font-size:13px; color:var(--cyan); word-break:break-all; line-height:1.45; cursor:pointer; text-shadow:0 0 10px rgba(0,200,255,.4); }
#cf-url:hover { text-decoration:underline; }
.cf-hint { font-size:10px; color:var(--text3); margin-top:4px; }
.gpu-row { margin-top:10px; font-size:12px; color:var(--text2); }
.gpu-row b { color:var(--text); }

/* ── Log panel ── */
#log-hdr {
  display:flex; align-items:center; gap:8px; flex-wrap:wrap;
  padding:9px 16px; flex-shrink:0;
  background:var(--bg2); border-bottom:2px solid var(--border);
}
.log-hdr-title { font-family:var(--hud); font-size:12px; font-weight:600; letter-spacing:.12em; color:var(--text); }
.flt {
  font-family:var(--hud); font-size:10px; letter-spacing:.1em;
  padding:4px 12px; border-radius:4px; cursor:pointer;
  background:transparent; border:1px solid rgba(255,255,255,.1);
  color:var(--text2); transition:all .15s; text-transform:uppercase;
}
.flt:hover  { border-color:var(--cyan); color:var(--cyan); }
.flt.active { background:var(--cyan-dim); border-color:var(--cyan); color:var(--cyan); }
#log-count { font-size:11px; color:var(--text2); margin-left:auto; }
.follow-label { display:flex; align-items:center; gap:6px; font-size:11px; color:var(--text2); cursor:pointer; }
.follow-label input { accent-color:var(--cyan); width:13px; height:13px; cursor:pointer; }
.btn-clear {
  font-family:var(--hud); font-size:10px; letter-spacing:.1em;
  padding:4px 12px; border-radius:4px; cursor:pointer;
  background:var(--red-dim); border:1px solid rgba(255,82,82,.4);
  color:var(--red); text-transform:uppercase; transition:all .15s;
}
.btn-clear:hover { background:rgba(255,82,82,.2); }

#log-body { flex:1; overflow-y:auto; padding:2px 0; font-size:13px; background:var(--bg); }
#log-body::-webkit-scrollbar { width:5px; }
#log-body::-webkit-scrollbar-track { background:transparent; }
#log-body::-webkit-scrollbar-thumb { background:var(--bg3); border-radius:3px; }

.ll {
  display:grid; grid-template-columns:76px 88px 1fr; gap:0 8px;
  padding:3px 16px; border-bottom:1px solid rgba(255,255,255,.03);
  animation:slideIn .12s ease;
}
.ll:hover { background:rgba(255,255,255,.03); }
@keyframes slideIn { from{opacity:0;transform:translateX(-4px)} to{opacity:1;transform:none} }

.ll-ts  { color:var(--text3); font-size:11px; padding-top:1px; white-space:nowrap; }
.ll-src { font-size:10px; font-weight:600; letter-spacing:.06em; padding:1px 5px; border-radius:3px; text-align:center; white-space:nowrap; align-self:start; }
.src-api         { color:var(--cyan);   background:var(--cyan-dim); }
.src-cloudflared { color:var(--purple); background:rgba(192,96,255,.1); }
.src-system      { color:var(--amber);  background:var(--amber-dim); }
.ll-txt { color:var(--text2); word-break:break-word; line-height:1.5; }
.lv-ok    .ll-txt { color:#4ddb8a; }
.lv-warn  .ll-txt { color:var(--amber); }
.lv-error .ll-txt { color:var(--red); font-weight:600; }
.lv-info  .ll-txt { color:var(--text2); }

/* ── Toast ── */
#toast {
  position:fixed; bottom:20px; right:20px; z-index:9999;
  background:var(--green-dim); border:1px solid var(--green);
  color:var(--green); font-family:var(--hud); font-size:12px; letter-spacing:.12em;
  padding:9px 20px; border-radius:6px;
  opacity:0; transform:translateY(8px); pointer-events:none; transition:all .25s;
}
#toast.show { opacity:1; transform:none; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.45} }
</style>
</head>
<body>
<div id="app">

  <!-- HEADER -->
  <div id="hdr">
    <div class="logo">MEMORA <span>:: SYSCTRL</span></div>
    <div id="hdr-clock">--:--:--</div>
    <div style="flex:1"></div>
    <div class="ws-pill">
      <div id="ws-dot"></div>
      <span id="ws-lbl">OFFLINE</span>
    </div>
  </div>

  <!-- VITALS STRIP — full width across top -->
  <div id="vitals-strip">
    <div class="ptitle">System Vitals</div>
    <div id="stat-grid">
      <div class="gauge gc">
        <div class="g-lbl">CPU</div>
        <div class="g-num" id="v-cpu">--</div>
        <div class="g-sub" id="s-cpu">utilization</div>
        <div class="g-bar"><div class="g-fill" id="b-cpu" style="width:0"></div></div>
        <canvas class="g-spark" id="sp-cpu"></canvas>
      </div>
      <div class="gauge gr">
        <div class="g-lbl">RAM</div>
        <div class="g-num" id="v-ram">--</div>
        <div class="g-sub" id="s-ram">% used</div>
        <div class="g-bar"><div class="g-fill" id="b-ram" style="width:0"></div></div>
        <canvas class="g-spark" id="sp-ram"></canvas>
      </div>
      <div class="gauge gd">
        <div class="g-lbl">DISK</div>
        <div class="g-num" id="v-dsk">--</div>
        <div class="g-sub" id="s-dsk">% used</div>
        <div class="g-bar"><div class="g-fill" id="b-dsk" style="width:0"></div></div>
        <canvas class="g-spark" id="sp-dsk"></canvas>
      </div>
      <div class="gauge gg">
        <div class="g-lbl">GPU</div>
        <div class="g-num" id="v-gpu">--</div>
        <div class="g-sub" id="s-gpu">load %</div>
        <div class="g-bar"><div class="g-fill" id="b-gpu" style="width:0"></div></div>
        <canvas class="g-spark" id="sp-gpu"></canvas>
      </div>
      <div class="gauge gv">
        <div class="g-lbl">VRAM</div>
        <div class="g-num" id="v-vrm">--</div>
        <div class="g-sub" id="s-vrm">MB used</div>
        <div class="g-bar"><div class="g-fill" id="b-vrm" style="width:0"></div></div>
        <canvas class="g-spark" id="sp-vrm"></canvas>
      </div>
      <div class="gauge gt">
        <div class="g-lbl">TEMP</div>
        <div class="g-num" id="v-tmp">--</div>
        <div class="g-sub" id="s-tmp">°C</div>
        <div class="g-bar"><div class="g-fill" id="b-tmp" style="width:0"></div></div>
        <canvas class="g-spark" id="sp-tmp"></canvas>
      </div>
    </div>
  </div>

  <!-- BOTTOM: left controls + right logs -->
  <div id="bottom">

    <!-- LEFT: process control + network -->
    <div id="left">

      <div class="panel">
        <div class="ptitle">Process Control</div>

        <!-- API -->
        <div class="proc-card">
          <div class="proc-top">
            <div class="proc-dot stopped" id="dot-api"></div>
            <div class="proc-name">api.py</div>
            <span class="proc-badge pb-stop" id="badge-api">STOPPED</span>
          </div>
          <div class="proc-meta">
            <span id="up-api">uptime: --</span>
            <span id="pid-api"></span>
            <span id="cpu-api">cpu: --</span>
            <span id="mem-api">mem: --</span>
          </div>
          <div class="proc-btns">
            <button class="btn btn-go"  onclick="cmd('/api/start')">Start</button>
            <button class="btn btn-stp" onclick="cmd('/api/stop')">Stop</button>
            <button class="btn btn-rst" onclick="cmd('/api/restart')">Restart</button>
          </div>
        </div>

        <!-- Cloudflared -->
        <div class="proc-card">
          <div class="proc-top">
            <div class="proc-dot stopped" id="dot-cf"></div>
            <div class="proc-name">cloudflared</div>
            <span class="proc-badge pb-stop" id="badge-cf">STOPPED</span>
          </div>
          <div class="proc-meta">
            <span id="up-cf">uptime: --</span>
            <span id="pid-cf"></span>
            <span id="cpu-cf">cpu: --</span>
            <span id="mem-cf">mem: --</span>
          </div>
          <div class="proc-btns">
            <button class="btn btn-go"  onclick="cmd('/cf/start')">Start</button>
            <button class="btn btn-stp" onclick="cmd('/cf/stop')">Stop</button>
            <button class="btn btn-rst" onclick="cmd('/cf/restart')">Restart</button>
          </div>
        </div>

        <!-- Tunnel URL -->
        <div id="cf-box">
          <div class="cf-lbl">Cloudflare Tunnel URL</div>
          <div id="cf-url" onclick="copyUrl()">Waiting for cloudflared...</div>
          <div class="cf-hint">Click to copy</div>
        </div>

        <div class="gpu-row">GPU: <b id="gpu-name">detecting...</b></div>
      </div>

      <!-- Network -->
      <div class="panel">
        <div class="ptitle">Network I/O</div>
        <div class="net-row">
          <div class="net-card">
            <div class="net-lbl">DOWNLOAD</div>
            <div class="net-val" id="net-rx">-- KB/s</div>
          </div>
          <div class="net-card">
            <div class="net-lbl">UPLOAD</div>
            <div class="net-val" id="net-tx">-- KB/s</div>
          </div>
        </div>
      </div>

    </div><!-- /left -->

    <!-- RIGHT: logs -->
    <div id="right">
      <div id="log-hdr">
        <span class="log-hdr-title">LOG STREAM</span>
        <button class="flt active" onclick="setFilter('all',this)">All</button>
        <button class="flt"        onclick="setFilter('api',this)">API</button>
        <button class="flt"        onclick="setFilter('cloudflared',this)">Tunnel</button>
        <button class="flt"        onclick="setFilter('system',this)">System</button>
        <span id="log-count">0 lines</span>
        <label class="follow-label">
          <input type="checkbox" id="follow" checked> Follow
        </label>
        <button class="btn-clear" onclick="clearLog()">Clear</button>
      </div>
      <div id="log-body"></div>
    </div>

  </div><!-- /bottom -->
</div>

<div id="toast">URL COPIED</div>

<script>
// ── State ──
let filter  = 'all';
let lines   = [];
let cfUrl   = null;
const MAX   = 1000;
const HIST  = 60;
const sparks = { cpu:[], ram:[], dsk:[], gpu:[], vrm:[], tmp:[] };

const SPARK_COL = {
  cpu:'#00c8ff', ram:'#00e5c8', dsk:'#c060ff',
  gpu:'#00e676', vrm:'#ffb300', tmp:'#ff5252'
};

// ── Clock ──
setInterval(() => {
  const n = new Date();
  document.getElementById('hdr-clock').textContent =
    n.toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
}, 1000);

// ── WebSocket ──
let sock, retryTimer;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  sock = new WebSocket(`${proto}://${location.host}/ws`);
  sock.onopen  = () => {
    document.getElementById('ws-dot').className = 'on';
    document.getElementById('ws-lbl').textContent = 'LIVE';
    sysLog('Monitor connected', 'ok');
    clearTimeout(retryTimer);
  };
  sock.onclose = () => {
    document.getElementById('ws-dot').className = '';
    document.getElementById('ws-lbl').textContent = 'OFFLINE';
    sysLog('Disconnected — retrying in 3 s...', 'warn');
    retryTimer = setTimeout(connect, 3000);
  };
  sock.onerror = () => sock.close();
  sock.onmessage = e => { try { dispatch(JSON.parse(e.data)); } catch(_){} };
}
connect();

// ── Dispatch ──
function dispatch(m) {
  if      (m.type === 'init')   { setProc('api', m.api); setProc('cf', m.cf); if (m.cf?.url) setUrl(m.cf.url); }
  else if (m.type === 'stats')  { updateStats(m); }
  else if (m.type === 'proc')   { const k = m.name === 'cloudflared' ? 'cf' : m.name; setProc(k, m); }
  else if (m.type === 'log')    { addLine(m); }
  else if (m.type === 'cf_url') { m.url ? setUrl(m.url) : clearUrl(); }
}

// ── Stats ──
function gauge(id, numText, pct, subText) {
  const n = document.getElementById('v-'+id); if (n) n.textContent = numText;
  const b = document.getElementById('b-'+id); if (b) b.style.width = Math.min(pct,100)+'%';
  const s = document.getElementById('s-'+id); if (s && subText !== undefined) s.textContent = subText;
}
function updateStats(s) {
  gauge('cpu', s.cpu_pct+'%',  s.cpu_pct,  s.cpu_pct+'% utilization');
  gauge('ram', s.ram_pct+'%',  s.ram_pct,  s.ram_used+' / '+s.ram_total+' GB');
  gauge('dsk', s.disk_pct+'%',s.disk_pct, s.disk_free+' GB free');

  if (s.gpu) {
    gauge('gpu', s.gpu.load+'%',  s.gpu.load, 'load %');
    const vp = (s.gpu.mem_used/s.gpu.mem_total)*100;
    gauge('vrm', s.gpu.mem_used+'',vp, s.gpu.mem_used+' / '+s.gpu.mem_total+' MB');
    gauge('tmp', s.gpu.temp+'',    Math.min((s.gpu.temp/90)*100,100), s.gpu.temp+' degrees C');
    document.getElementById('gpu-name').textContent = s.gpu.name;
  }

  document.getElementById('net-rx').textContent = s.rx_kb+' KB/s';
  document.getElementById('net-tx').textContent = s.tx_kb+' KB/s';

  if (s.api_proc?.cpu   !== undefined) document.getElementById('cpu-api').textContent = 'cpu: '+s.api_proc.cpu+'%';
  if (s.api_proc?.mem_mb!== undefined) document.getElementById('mem-api').textContent = 'mem: '+s.api_proc.mem_mb+' MB';
  if (s.cf_proc?.cpu    !== undefined) document.getElementById('cpu-cf').textContent  = 'cpu: '+s.cf_proc.cpu+'%';
  if (s.cf_proc?.mem_mb !== undefined) document.getElementById('mem-cf').textContent  = 'mem: '+s.cf_proc.mem_mb+' MB';
  if (s.api_uptime != null) document.getElementById('up-api').textContent = 'uptime: '+fmtUp(s.api_uptime);
  if (s.cf_uptime  != null) document.getElementById('up-cf').textContent  = 'uptime: '+fmtUp(s.cf_uptime);

  push('cpu', s.cpu_pct);
  push('ram', s.ram_pct);
  push('dsk', s.disk_pct);
  push('gpu', s.gpu ? s.gpu.load : 0);
  push('vrm', s.gpu ? (s.gpu.mem_used/s.gpu.mem_total)*100 : 0);
  push('tmp', s.gpu ? Math.min((s.gpu.temp/90)*100,100) : 0);
}

function fmtUp(s) {
  if (s < 60)   return s+'s';
  if (s < 3600) return Math.floor(s/60)+'m '+s%60+'s';
  return Math.floor(s/3600)+'h '+Math.floor((s%3600)/60)+'m';
}

// ── Sparklines ──
function push(k, v) {
  const a = sparks[k]; a.push(v);
  if (a.length > HIST) a.shift();
  drawSpark(k, a);
}
function drawSpark(k, a) {
  const c = document.getElementById('sp-'+k);
  if (!c || a.length < 2) return;
  const W = c.offsetWidth || 90, H = c.offsetHeight || 34;
  c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  ctx.clearRect(0,0,W,H);
  const step = W / (a.length-1);
  ctx.beginPath();
  a.forEach((v,i) => {
    const x = i*step, y = H - (v/100)*H*.9 - 2;
    i === 0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.strokeStyle = SPARK_COL[k];
  ctx.lineWidth   = 2;
  ctx.stroke();
  // area fill
  ctx.lineTo((a.length-1)*step, H); ctx.lineTo(0,H); ctx.closePath();
  const g = ctx.createLinearGradient(0,0,0,H);
  g.addColorStop(0, SPARK_COL[k]+'44');
  g.addColorStop(1,'transparent');
  ctx.fillStyle = g; ctx.fill();
}

// ── Process status ──
function setProc(key, data) {
  if (!data) return;
  const st = data.status || 'stopped';
  const dot   = document.getElementById('dot-'+key);
  const badge = document.getElementById('badge-'+key);
  if (dot)   { dot.className = 'proc-dot '+st; }
  if (badge) {
    badge.textContent = st.toUpperCase();
    badge.className   = 'proc-badge '+(st==='running'?'pb-run':st==='error'?'pb-err':'pb-stop');
  }
  if (data.pid) {
    const p = document.getElementById('pid-'+key);
    if (p) p.textContent = 'PID '+data.pid;
  }
}

// ── CF URL ──
function setUrl(url) {
  cfUrl = url;
  const el = document.getElementById('cf-url');
  el.textContent = url;
  el.style.color = 'var(--cyan)';
}
function clearUrl() {
  cfUrl = null;
  document.getElementById('cf-url').textContent = 'Waiting for cloudflared...';
}
function copyUrl() {
  if (!cfUrl) return;
  navigator.clipboard.writeText(cfUrl).then(() => {
    const t = document.getElementById('toast');
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2000);
  });
}

// ── Logs ──
function sysLog(text, level='info') { addLine({src:'system',level,text,ts:Date.now()/1000}); }

function fmtTs(ts) {
  const d = new Date(ts*1000);
  return d.toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
}

function addLine(m) {
  lines.push(m);
  if (lines.length > MAX+50) lines = lines.slice(-MAX);
  if (filter === 'all' || filter === m.src) renderLine(m);
  document.getElementById('log-count').textContent = lines.length+' lines';
}

function renderLine(m) {
  const el = document.getElementById('log-body');
  const d  = document.createElement('div');
  d.className = 'll lv-'+(m.level||'info');
  d.innerHTML =
    `<span class="ll-ts">${fmtTs(m.ts)}</span>`+
    `<span class="ll-src src-${m.src||'system'}">${(m.src||'sys').toUpperCase()}</span>`+
    `<span class="ll-txt">${esc(m.text)}</span>`;
  el.appendChild(d);
  const f = document.getElementById('follow');
  if (f && f.checked) el.scrollTop = el.scrollHeight;
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function setFilter(src, btn) {
  filter = src;
  document.querySelectorAll('.flt').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const el = document.getElementById('log-body');
  el.innerHTML = '';
  (src==='all' ? lines : lines.filter(l=>l.src===src)).forEach(renderLine);
}

function clearLog() {
  lines = [];
  document.getElementById('log-body').innerHTML = '';
  document.getElementById('log-count').textContent = '0 lines';
}

// ── Commands ──
async function cmd(path) {
  try {
    const r = await fetch(path, {method:'POST'});
    const d = await r.json();
    sysLog(path+' -> '+JSON.stringify(d), 'ok');
  } catch(e) { sysLog('Command failed: '+e, 'error'); }
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
    print("\n" + "=" * 55)
    print("  MEMORA :: SYSCTRL MONITOR")
    print("=" * 55)
    print(f"  Local:   http://localhost:{MONITOR_PORT}")
    print(f"  Network: http://{ip}:{MONITOR_PORT}")
    print(f"  API dir: {BASE_DIR}")
    print("=" * 55 + "\n")
    uvicorn.run("monitor:app", host="0.0.0.0", port=MONITOR_PORT,
                reload=False, log_level="warning")

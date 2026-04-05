const { app, BrowserWindow, ipcMain, desktopCapturer, screen } = require('electron');
const path = require('path');
const fs   = require('fs');
const { spawn } = require('child_process');

let backendProcess = null;

// Fix GPU shader cache conflict on Windows
app.commandLine.appendSwitch('disable-gpu-shader-disk-cache');
app.commandLine.appendSwitch('disable-dev-shm-usage');

let win, fabWin;

// ── Main overlay window (hidden by default) ──────────────
function createMainWindow() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;
  const W = 290, H = 420;

  win = new BrowserWindow({
    width:  W,
    height: H,
    x: width - W - 20,
    y: Math.round((height - H) / 2),
    frame: false,
    transparent: true,
    roundedCorners: true,
    hasShadow: false,
    backgroundColor: '#00000000',
    alwaysOnTop: true,
    resizable: false,
    skipTaskbar: false,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.webContents.session.setPermissionRequestHandler((wc, permission, callback) => {
    callback(['media', 'display-capture', 'screen'].includes(permission));
  });

  win.webContents.session.setDisplayMediaRequestHandler((request, callback) => {
    desktopCapturer.getSources({ types: ['screen'] }).then(sources => {
      callback({ video: sources[0] });
    });
  });

  win.loadFile(path.join(__dirname, 'deepfake_detector_ui.html'));
  win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  // On close button inside the app → hide, don't quit
  win.on('close', e => {
    if (!app.isQuitting) {
      e.preventDefault();
      win.hide();
    }
  });
}

// ── FAB (always-visible launcher) ───────────────────────
function createFAB() {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;

  fabWin = new BrowserWindow({
    width:  62,
    height: 62,
    x: width  - 80,
    y: height - 90,
    frame: false,
    transparent: true,
    roundedCorners: true,
    hasShadow: false,
    backgroundColor: '#00000000',
    alwaysOnTop: true,
    skipTaskbar: true,
    resizable: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  fabWin.loadFile(path.join(__dirname, 'fab.html'));
  fabWin.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
}

app.whenReady().then(() => {
  const backendPath = path.join(__dirname, '..', 'backend', 'api.py');
  backendProcess = spawn('python', [backendPath], {
    cwd: path.join(__dirname, '..', 'backend'),
    stdio: 'ignore'
  });

  createMainWindow();
  createFAB();
});

app.on('before-quit', () => { app.isQuitting = true; });
app.on('will-quit', () => {
  if (backendProcess) backendProcess.kill();
});
app.on('window-all-closed', () => app.quit());

// ── IPC: toggle main overlay ─────────────────────────────────
ipcMain.on('toggle-main', () => {
  if (!win) return;
  if (win.isVisible()) {
    win.hide();
    return;
  }

  // Position main window next to the FAB
  const [fx, fy]  = fabWin.getPosition();
  const fabW      = 62;
  const fabH      = 62;
  const W         = 290;
  const H         = 420; // Matches window creation size
  const { width: sw, height: sh } = screen.getPrimaryDisplay().workAreaSize;
  const margin    = 10;

  // Decide which side to open on (left or right of FAB)
  let x = fx - W - margin;
  if (x < 0) x = fx + fabW + margin;   // not enough space on left → open right
  x = Math.max(0, Math.min(x, sw - W));

  // Align vertically: bottom-align with FAB, clamp to screen
  let y = fy + fabH - H;
  y = Math.max(0, Math.min(y, sh - H));

  win.setPosition(Math.round(x), Math.round(y));
  win.show();
  win.focus();
});

// ── IPC: close main (hides it) ───────────────────────────
ipcMain.on('close-app', () => {
  if (win) win.hide();
});

// ── IPC: minimize main ───────────────────────────────────
ipcMain.on('minimize-app', () => {
  if (win) win.minimize();
});

// ── IPC: drag main window ────────────────────────────────
ipcMain.on('drag-window', (_, { dx, dy }) => {
  if (!win) return;
  const [x, y] = win.getPosition();
  win.setPosition(x + dx, y + dy);
});

// ── IPC: drag FAB ────────────────────────────────────────
ipcMain.on('drag-fab', (_, { dx, dy }) => {
  if (!fabWin) return;
  const [x, y] = fabWin.getPosition();
  fabWin.setPosition(x + dx, y + dy);
});

// ── IPC: screen sources ──────────────────────────────────
ipcMain.handle('get-sources', async () => {
  const sources = await desktopCapturer.getSources({
    types: ['screen', 'window'],
    thumbnailSize: { width: 1, height: 1 },
  });
  return sources.map(s => ({ id: s.id, name: s.name }));
});

// ── IPC: save video ──────────────────────────────────────
ipcMain.handle('save-video', async (_, buffer) => {
  const dir = path.join(__dirname, '..', 'backend', 'saved_videos');
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  const p = path.join(dir, `capture_${Date.now()}.webm`);
  fs.writeFileSync(p, Buffer.from(buffer));
  return p;
});

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // Main window
  getSources:  ()       => ipcRenderer.invoke('get-sources'),
  saveVideo:   (buffer) => ipcRenderer.invoke('save-video', buffer),
  closeApp:    ()       => ipcRenderer.send('close-app'),
  minimizeApp: ()       => ipcRenderer.send('minimize-app'),
  dragWindow:  (delta)  => ipcRenderer.send('drag-window', delta),
  // FAB
  toggleMain:  ()       => ipcRenderer.send('toggle-main'),
  dragFab:     (delta)  => ipcRenderer.send('drag-fab', delta),
});

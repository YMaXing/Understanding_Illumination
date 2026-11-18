# Chrome DevTools Protocol driver for the annotation tool's browser tests (Windows PowerShell 5, .NET ClientWebSocket;
# no extra dependencies). WSL (NAT networking) cannot reach Chrome's debugging port, so this runs on the Windows side:
# selftest.py writes a JSON list of steps, this script launches its own headless Chrome (own throwaway profile; it
# never touches the user's browser), runs the steps with REAL input events, writes the results and stops that Chrome.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File cdp_driver.ps1 -Steps steps.json -Out results.json -Profile DIR
#
# Steps (objects, run in order; any failure stops the run and is reported):
#   {"nav": url}                      Page.navigate, then wait for document.readyState == complete
#   {"wait": js, "timeout_ms": N}     poll Runtime.evaluate until truthy
#   {"eval": js, "label": L}          evaluate (promises awaited), keep the value under L
#   {"type": text}                    Input.insertText (real text input into the focused element)
#   {"key": name, "mods": n}          Input.dispatchKeyEvent keyDown (+char) + keyUp; mods: 1 Alt, 2 Ctrl, 4 Meta, 8 Shift
#   {"click_js": js}                  js returns [clientX, clientY]; mouseMoved, mousePressed, mouseReleased (left)
#   {"drag_js": js}                   js returns [x0, y0, x1, y1]; press at 0, move in 5 steps, release at 1
#   {"sleep": ms}
#   {"move_js": js}                   js returns [clientX, clientY]; mouseMoved
#   {"shot": path_windows, "format": "png"|"jpeg", "quality": q, "clip_js": js -> [x, y, w, h]}   Page.captureScreenshot
param([string]$Steps, [string]$Out, [string]$Profile, [int]$Port = 9333, [int]$Width = 1600, [int]$Height = 1000,
      [string]$Chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe")
$ErrorActionPreference = "Stop"
$ct = [System.Threading.CancellationToken]::None
$results = New-Object System.Collections.ArrayList
$proc = Start-Process -FilePath $Chrome -PassThru -ArgumentList @("--headless=new", "--disable-gpu", "--no-first-run",
  "--no-default-browser-check", "--remote-debugging-port=$Port", "--remote-allow-origins=*", "--user-data-dir=`"$Profile`"",
  "--window-size=$Width,$Height", "about:blank")
$script:id = 0
function Send-Cdp($ws, $method, $params) {
  $script:id++
  $msg = @{id = $script:id; method = $method; params = $params} | ConvertTo-Json -Depth 20 -Compress
  $b = [Text.Encoding]::UTF8.GetBytes($msg)
  $ws.SendAsync([ArraySegment[byte]]::new($b), [System.Net.WebSockets.WebSocketMessageType]::Text, $true, $ct).Wait()
  while ($true) {
    $ms = New-Object System.IO.MemoryStream
    do {
      $buf = New-Object byte[] 1048576
      $r = $ws.ReceiveAsync([ArraySegment[byte]]::new($buf), $ct).Result
      $ms.Write($buf, 0, $r.Count)
    } while (-not $r.EndOfMessage)
    $o = [Text.Encoding]::UTF8.GetString($ms.ToArray()) | ConvertFrom-Json
    if ($o.id -eq $script:id) {
      if ($o.error) { throw "CDP $method error: $($o.error.message)" }
      return $o.result
    }
  }
}
function Eval-Js($ws, $js) {
  $r = Send-Cdp $ws "Runtime.evaluate" @{expression = $js; returnByValue = $true; awaitPromise = $true}
  if ($r.exceptionDetails) { throw "JS exception: $($r.exceptionDetails.exception.description)" }
  return $r.result.value
}
$KEYS = @{ "Enter" = @(13, "Enter", "`r"); "Escape" = @(27, "Escape", $null); "Delete" = @(46, "Delete", $null);
  "Backspace" = @(8, "Backspace", $null); "ArrowLeft" = @(37, "ArrowLeft", $null); "ArrowRight" = @(39, "ArrowRight", $null);
  "ArrowUp" = @(38, "ArrowUp", $null); "ArrowDown" = @(40, "ArrowDown", $null); "Tab" = @(9, "Tab", $null);
  "[" = @(219, "BracketLeft", "["); "]" = @(221, "BracketRight", "]"); "?" = @(191, "Slash", "?"); " " = @(32, "Space", " ") }
function Press-Key($ws, $k, $mods) {
  if ($KEYS.ContainsKey($k)) { $vk, $code, $text = $KEYS[$k] }
  elseif ($k.Length -eq 1 -and $k -match "[a-z]") { $vk = [int][char]$k.ToUpper(); $code = "Key" + $k.ToUpper(); $text = $k }
  elseif ($k.Length -eq 1 -and $k -match "[A-Z]") { $vk = [int][char]$k; $code = "Key" + $k; $text = $k; $mods = $mods -bor 8 }
  elseif ($k.Length -eq 1 -and $k -match "[0-9]") { $vk = [int][char]$k; $code = "Digit" + $k; $text = $k }
  else { throw "unknown key $k" }
  $down = @{type = $(if ($text) { "keyDown" } else { "rawKeyDown" }); key = $k; code = $code; windowsVirtualKeyCode = $vk;
            nativeVirtualKeyCode = $vk; modifiers = [int]$mods}
  if ($text -and -not ($mods -band 7)) { $down.text = $text; $down.unmodifiedText = $text }
  Send-Cdp $ws "Input.dispatchKeyEvent" $down | Out-Null
  Send-Cdp $ws "Input.dispatchKeyEvent" @{type = "keyUp"; key = $k; code = $code; windowsVirtualKeyCode = $vk;
                                           nativeVirtualKeyCode = $vk; modifiers = [int]$mods} | Out-Null
}
function Mouse($ws, $type, $x, $y, $buttons) {
  $p = @{type = $type; x = [double]$x; y = [double]$y; button = "left"; buttons = $buttons; clickCount = 1}
  if ($type -eq "mouseMoved") { $p.button = $(if ($buttons) { "left" } else { "none" }) }
  Send-Cdp $ws "Input.dispatchMouseEvent" $p | Out-Null
}
$ws = $null
try {
  $deadline = (Get-Date).AddSeconds(30)
  while ($true) {
    try { $list = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/json/list" -TimeoutSec 2; if ($list) { break } } catch {}
    if ((Get-Date) -gt $deadline) { throw "Chrome debugging port $Port did not open" }
    Start-Sleep -Milliseconds 300
  }
  $page = @($list | Where-Object { $_.type -eq "page" })[0]
  $ws = New-Object System.Net.WebSockets.ClientWebSocket
  $ws.Options.KeepAliveInterval = [TimeSpan]::FromSeconds(30)
  $ws.ConnectAsync([Uri]$page.webSocketDebuggerUrl, $ct).Wait()
  Send-Cdp $ws "Page.enable" @{} | Out-Null
  Send-Cdp $ws "Emulation.setDeviceMetricsOverride" @{width = $Width; height = $Height; deviceScaleFactor = 1; mobile = $false} | Out-Null
  $plan = Get-Content -Raw -Encoding UTF8 $Steps | ConvertFrom-Json    # not $steps: PowerShell names are case-insensitive
  $n = 0
  foreach ($s in $plan) {
    $n++
    $rec = @{n = $n; step = ($s | ConvertTo-Json -Compress -Depth 5)}
    if ($s.nav) {
      Send-Cdp $ws "Page.navigate" @{url = $s.nav} | Out-Null
      $t0 = Get-Date
      while ((Eval-Js $ws "document.readyState") -ne "complete") { if (((Get-Date) - $t0).TotalSeconds -gt 30) { throw "page load timeout" }; Start-Sleep -Milliseconds 100 }
    } elseif ($s.wait) {
      $to = $(if ($s.timeout_ms) { $s.timeout_ms } else { 15000 }); $t0 = Get-Date
      while (-not (Eval-Js $ws $s.wait)) {
        if (((Get-Date) - $t0).TotalMilliseconds -gt $to) { throw "timeout waiting for: $($s.wait)" }
        Start-Sleep -Milliseconds 50
      }
    } elseif ($s.eval) { $rec.value = Eval-Js $ws $s.eval
    } elseif ($null -ne $s.type) { Send-Cdp $ws "Input.insertText" @{text = [string]$s.type} | Out-Null
    } elseif ($s.key) { Press-Key $ws $s.key $(if ($s.mods) { $s.mods } else { 0 })
    } elseif ($s.click_js) {
      $xy = Eval-Js $ws $s.click_js
      Mouse $ws "mouseMoved" $xy[0] $xy[1] 0; Mouse $ws "mousePressed" $xy[0] $xy[1] 1; Mouse $ws "mouseReleased" $xy[0] $xy[1] 0
      $rec.value = $xy
    } elseif ($s.drag_js) {
      $q = Eval-Js $ws $s.drag_js
      Mouse $ws "mouseMoved" $q[0] $q[1] 0; Mouse $ws "mousePressed" $q[0] $q[1] 1
      for ($k = 1; $k -le 5; $k++) { Mouse $ws "mouseMoved" ($q[0] + ($q[2] - $q[0]) * $k / 5) ($q[1] + ($q[3] - $q[1]) * $k / 5) 1 }
      Mouse $ws "mouseReleased" $q[2] $q[3] 0
    } elseif ($s.move_js) {
      $xy = Eval-Js $ws $s.move_js
      Mouse $ws "mouseMoved" $xy[0] $xy[1] 0
    } elseif ($s.sleep) { Start-Sleep -Milliseconds $s.sleep
    } elseif ($s.shot) {
      $prm = @{format = $(if ($s.format) { $s.format } else { "png" })}
      if ($s.quality) { $prm.quality = [int]$s.quality }
      if ($s.clip_js) { $c = Eval-Js $ws $s.clip_js; $prm.clip = @{x = [double]$c[0]; y = [double]$c[1]; width = [double]$c[2]; height = [double]$c[3]; scale = 1} }
      $img = Send-Cdp $ws "Page.captureScreenshot" $prm
      [IO.File]::WriteAllBytes($s.shot, [Convert]::FromBase64String($img.data))
    }
    if ($s.label) { $rec.label = $s.label }
    [void]$results.Add($rec)
  }
  $status = "done"
} catch {
  $status = "error: " + $_.Exception.Message + " (at step $n)"
} finally {
  if ($ws) { try { $ws.Dispose() } catch {} }
  try { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue } catch {}
}
@{status = $status; results = $results} | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 $Out

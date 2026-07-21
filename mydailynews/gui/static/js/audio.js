import { byId, setStatus } from "./dom.js";

const SEEK_SECONDS = 10;
const SPEED_STEP = 0.25;
const MIN_SPEED = 0.5;
const MAX_SPEED = 2;

let audio = null;
let nodes = null;
let currentUrl = "";
let currentTitle = "Audio";
let collapsed = true;

export function initAudioPlayer() {
  audio = new Audio();
  nodes = {
    player: byId("audioPlayer"),
    toggle: byId("toggleAudioPlayer"),
    title: byId("audioTitle"),
    stop: byId("audioStopButton"),
    rewind: byId("audioRewindButton"),
    playPause: byId("audioPlayPauseButton"),
    forward: byId("audioForwardButton"),
    seek: byId("audioSeek"),
    time: byId("audioTime"),
    slow: byId("audioSlowButton"),
    speed: byId("audioSpeedLabel"),
    fast: byId("audioFastButton"),
  };
  if (!nodes.player) {
    return;
  }

  ["play", "pause", "ended", "timeupdate", "durationchange", "loadedmetadata", "ratechange"].forEach((eventName) => {
    audio.addEventListener(eventName, updateAudioUi);
  });
  nodes.toggle.addEventListener("click", () => setCollapsed(!collapsed));
  nodes.stop.addEventListener("click", stopAudio);
  nodes.rewind.addEventListener("click", () => seekBy(-SEEK_SECONDS));
  nodes.playPause.addEventListener("click", toggleCurrentAudio);
  nodes.forward.addEventListener("click", () => seekBy(SEEK_SECONDS));
  nodes.seek.addEventListener("input", () => {
    if (Number.isFinite(audio.duration)) {
      audio.currentTime = Number(nodes.seek.value || 0);
    }
  });
  nodes.slow.addEventListener("click", () => changeSpeed(-SPEED_STEP));
  nodes.fast.addEventListener("click", () => changeSpeed(SPEED_STEP));
  setCollapsed(true);
  updateAudioUi();
}

export function toggleAudio(url, title = "Audio") {
  if (!url || !audio) {
    return;
  }

  const switched = loadAudio(url, title);
  showPlayer();
  if (!switched && !audio.paused) {
    audio.pause();
    return;
  }
  playAudio();
}

export function audioButtonLabel(url) {
  return currentUrl && currentUrl === audioUrl(url) && audio && !audio.paused ? "Pause" : "Play";
}

function loadAudio(url, title) {
  const absoluteUrl = audioUrl(url);
  if (currentUrl === absoluteUrl) {
    return false;
  }

  audio.pause();
  audio.src = url;
  currentUrl = absoluteUrl;
  currentTitle = title || "Audio";
  return true;
}

function showPlayer() {
  if (nodes.player.hidden) {
    nodes.player.hidden = false;
    setCollapsed(false);
  }
  updateAudioUi();
}

function setCollapsed(next) {
  collapsed = next;
  nodes.player.classList.toggle("collapsed", collapsed);
  nodes.toggle.setAttribute("aria-expanded", String(!collapsed));
  nodes.toggle.title = `${collapsed ? "Open" : "Close"} audio`;
  document.body.classList.toggle("audio-player-active", !nodes.player.hidden);
  document.body.classList.toggle("audio-player-expanded", !nodes.player.hidden && !collapsed);
}

function toggleCurrentAudio() {
  if (!currentUrl) {
    return;
  }
  if (audio.paused) {
    playAudio();
  } else {
    audio.pause();
  }
}

function playAudio() {
  audio.play().catch((error) => setStatus(error.message, true));
}

function stopAudio() {
  if (!currentUrl) {
    return;
  }
  audio.pause();
  audio.currentTime = 0;
  updateAudioUi();
}

function seekBy(seconds) {
  if (!currentUrl || !Number.isFinite(audio.duration)) {
    return;
  }
  audio.currentTime = Math.max(0, Math.min(audio.duration, audio.currentTime + seconds));
}

function changeSpeed(delta) {
  audio.playbackRate = Math.max(MIN_SPEED, Math.min(MAX_SPEED, audio.playbackRate + delta));
}

function updateAudioUi() {
  if (!nodes || !audio) {
    return;
  }

  const duration = Number.isFinite(audio.duration) ? audio.duration : 0;
  const current = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
  const hasAudio = Boolean(currentUrl);
  nodes.title.textContent = currentTitle;
  nodes.playPause.textContent = audio.paused ? "Play" : "Pause";
  nodes.time.textContent = `${formatTime(current)} / ${duration ? formatTime(duration) : "--:--"}`;
  nodes.seek.max = String(duration || 0);
  nodes.seek.value = String(duration ? Math.min(current, duration) : 0);
  nodes.seek.disabled = !duration;
  nodes.speed.textContent = `${Number(audio.playbackRate.toFixed(2))}x`;
  [nodes.stop, nodes.rewind, nodes.playPause, nodes.forward, nodes.slow, nodes.fast].forEach((node) => {
    node.disabled = !hasAudio;
  });
  updateReportAudioButton();
}

function updateReportAudioButton() {
  const button = byId("playReportAudio");
  if (button) {
    button.textContent = audioButtonLabel(button.dataset.audioUrl);
  }
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const remainder = String(total % 60).padStart(2, "0");
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${remainder}` : `${minutes}:${remainder}`;
}

function audioUrl(url) {
  return new URL(url, window.location.href).href;
}

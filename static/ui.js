document.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;

  if (btn.type === "submit") {
    sessionStorage.setItem("scrollY", window.scrollY.toString());
  }
});

window.addEventListener("load", () => {
  const y = sessionStorage.getItem("scrollY");
  if (!y) return;

  requestAnimationFrame(() => {
    window.scrollTo(0, parseInt(y, 10));
  });
});

const themes = {
  dark: "#0f1115",
  gray: "#1e1e1e",
  slate: "#1b2028",
  midnight: "#0a0c10",
};

const photoMap = {
  "photo-1": "/static/backgrounds/bg1.jpg",
  "photo-2": "/static/backgrounds/bg2.jpg",
  "photo-3": "/static/backgrounds/bg3.jpg",
};

function applyTheme(name) {
  document.body.classList.remove("bg-photo", "bg-gradient");
  document.documentElement.style.removeProperty("--bg-image");

  if (themes[name]) {
    document.documentElement.style.setProperty("--bg", themes[name]);
  }

  if (photoMap[name]) {
    document.documentElement.style.setProperty(
      "--bg-image",
      `url("${photoMap[name]}")`
    );
    document.body.classList.add("bg-photo");
  }

  if (name === "gradient") {
    document.body.classList.add("bg-gradient");
  }

  localStorage.setItem("ui_theme", name);
}

document.addEventListener("DOMContentLoaded", () => {

  const hud = document.getElementById("vram-hud");
  if (hud) {
    setInterval(async () => {
      try {
        const res = await fetch("/vram");
        const data = await res.json();

        if (data.text) {
          hud.textContent = data.text;
          hud.style.display = "block";
        } else {
          hud.style.display = "none";
        }
      } catch {
      }
    }, 10000);
  }

  const select = document.getElementById("theme-select");
  if (!select) return;

  const saved = localStorage.getItem("ui_theme") || "dark";
  select.value = saved;
  applyTheme(saved);

  select.addEventListener("change", (e) => {
    applyTheme(e.target.value);
  });
});

document.getElementById("openDatasetBtn").addEventListener("click", async () => {
  const params = new URLSearchParams(window.location.search);
  const project = params.get("project");

  try {
    const res = await fetch(`/api/open_dataset_folder?project=${encodeURIComponent(project)}`, {
      method: "POST"
    });

    if (!res.ok) {
      console.warn("Could not open dataset folder");
    }
  } catch (err) {
    console.error("Error opening dataset folder", err);
  }
});

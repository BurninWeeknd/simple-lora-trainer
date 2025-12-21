const datasetState = {
  images: [],
  selectedIndex: null,
  dirty: false,
  filter: ""
};

function resetDataset() {
        datasetState.root = null;
        datasetState.images = [];
        datasetState.selectedIndex = null;
        datasetState.dirty = false;

        console.log("[dataset] state reset", structuredClone(datasetState));
}

function renderImageList() {
        const listEl = document.getElementById("image-list");
        if (!listEl) return;

        const q = datasetState.filter.trim().toLowerCase();
        listEl.innerHTML = "";

        datasetState.images.forEach((img, index) => {
        if (q && !img.name.toLowerCase().includes(q)) return;

        const li = document.createElement("li");
        li.textContent = img.name;
        li.style.cursor = "pointer";
        li.style.padding = "4px 6px";
        li.style.borderRadius = "4px";

        if (index === datasetState.selectedIndex) {
        li.style.background = "var(--bg)";
        }

        li.addEventListener("click", () => {
        selectImage(index);
        renderImageList();
        li.scrollIntoView({ block: "nearest" });
        });

        listEl.appendChild(li);
        });
}

function renderImagePreview() {
        const previewEl = document.getElementById("image-preview");
        if (!previewEl) return;

        if (datasetState.selectedIndex === null) {
        previewEl.textContent = "No image selected";
        return;
        }

        const img = datasetState.images[datasetState.selectedIndex];

        previewEl.innerHTML = `
        <img
        src="/api/dataset/image/${encodeURIComponent(img.path)}"
        alt="${img.name}"
        style="
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 6px;
        "
        />
        `;
}

function renderCaptionEditor() {
        const textarea = document.getElementById("caption-editor");
        if (!textarea) return;

        if (datasetState.selectedIndex === null) {
        textarea.value = "";
        textarea.disabled = true;
        return;
        }

        const img = datasetState.images[datasetState.selectedIndex];
        textarea.disabled = false;
        textarea.value = img.caption || "";
}

function saveCaptions() {
        if (!datasetState.dirty) {
        console.log("[dataset] nothing to save");
        return;
        }

        fetch("/api/dataset/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        images: datasetState.images.map(img => ({
        name: img.name,
        caption: img.caption
        }))
        })
        })
        .then(res => res.json())
        .then(data => {
        console.log("[dataset] save response:", data);

        datasetState.images.forEach(img => {
        img.dirty = false;
        });
        datasetState.dirty = false;
        })
        .catch(err => {
        console.error("[dataset] save failed:", err);
        });
}

function loadImages(imageList) {
        datasetState.images = imageList.map(img => ({
        name: img.name,
        path: img.rel_path,
        caption: img.caption ?? "",
        dirty: false
        }));

        datasetState.selectedIndex = datasetState.images.length ? 0 : null;

        renderImageList();
        renderImagePreview();
        renderCaptionEditor();
}

function selectImage(index) {
        if (index < 0 || index >= datasetState.images.length) return;

        datasetState.selectedIndex = index;
        console.log("[dataset] selected image:", datasetState.images[index]);

        renderImageList();
        renderImagePreview();
        renderCaptionEditor();
}

function loadProjects() {
        fetch("/api/projects")
        .then(r => r.json())
        .then(data => {
        const sel = document.getElementById("project-select");
        sel.innerHTML = "";

        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "-- Select --";
        placeholder.disabled = true;
        placeholder.selected = true;
        sel.appendChild(placeholder);

        data.projects.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        sel.appendChild(opt);
        });
        })
        .catch(err => {
        console.error("[dataset] failed to load projects", err);
        });
}


function loadDatasetFromUI() {
        const project = document.getElementById("project-select").value;
        const datasetPath = document
        .getElementById("dataset-path-input")
        .value.trim();

        const errorBox = document.getElementById("dataset-error");
        errorBox.style.display = "none";

        if (!project || !datasetPath) {
        errorBox.textContent = "Project and dataset path are required";
        errorBox.style.display = "block";
        return;
        }

        fetch("/api/dataset/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        project,
        dataset_path: datasetPath,
        }),
        })
        .then(async res => {
        const data = await res.json();
        if (!res.ok) throw data;
        return data;
        })
        .then(data => {
        console.log("[dataset] loaded", data);
        loadImages(data.images);
        })
        .catch(err => {
        errorBox.textContent =
        err.error || "Failed to load dataset";
        errorBox.style.display = "block";
        });
}

function autoCaptionAll() {
const overwrite = document.getElementById("autocaption-overwrite").checked;

        const images = datasetState.images
        .filter(img => overwrite || !img.caption?.trim())
        .map(img => ({ name: img.name }));

        if (!images.length) {
        console.log("[dataset] no images to caption");
        return;
        }

        fetch("/api/dataset/autocaption", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        images,
        overwrite
        })
        })
        .then(r => r.json())
        .then(data => {
        data.captions.forEach(c => {
        const img = datasetState.images.find(i => i.name === c.name);
        if (img) {
        img.caption = c.caption;
        img.dirty = false;
        }
        });

        renderCaptionEditor();
        console.log("[dataset] auto-caption complete");
        })
        .catch(err => {
        console.error("[dataset] auto-caption failed", err);
        });
}

document.addEventListener("DOMContentLoaded", () => {
console.log("[dataset] Dataset Prep loaded");

        loadProjects();

        const projectSelect = document.getElementById("project-select");
        const datasetPathInput = document.getElementById("dataset-path-input");

        projectSelect.addEventListener("change", () => {
        const project = projectSelect.value;
                if (!project) return;

                fetch(`/api/project/config/${project}`)
                .then(r => r.json())
                .then(data => {
                datasetPathInput.value = data.dataset_path || "";
                })
                .catch(err => {
                console.error("[dataset] failed to load config", err);
                });
                });

        document
        .getElementById("load-dataset-btn")
        .addEventListener("click", loadDatasetFromUI);
        const filterInput = document.getElementById("image-filter");
                if (filterInput) {
                filterInput.addEventListener("input", (e) => {
                datasetState.filter = e.target.value || "";
                renderImageList();
                });
                }
        const captionEditor = document.getElementById("caption-editor");
                if (captionEditor) {
                captionEditor.addEventListener("input", (e) => {
                if (datasetState.selectedIndex === null) return;

        const img = datasetState.images[datasetState.selectedIndex];
                img.caption = e.target.value;
                img.dirty = true;
                datasetState.dirty = true;
                });
                }
        const autoBtn = document.getElementById("autocaption-btn");
                if (autoBtn) {
                autoBtn.addEventListener("click", autoCaptionAll);
                console.log("[dataset] autocaption button wired");
                }

        const saveBtn = document.getElementById("save-captions-btn");
                if (saveBtn) {
                saveBtn.addEventListener("click", saveCaptions);
                }
});

import { teams } from "./state.js";
import { getSavedTeams, setSavedTeams, persistCurrent } from "./storage.js";
import { renderAll } from "./render.js";

function teamNameInput(side) {
  return document.getElementById("teamName" + side);
}

export function renderSavedList() {
  const saved = getSavedTeams();
  const container = document.getElementById("savedList");
  container.innerHTML = "";
  const names = Object.keys(saved);
  if (names.length === 0) {
    container.innerHTML = `<span class="empty-note">No saved teams yet.</span>`;
    return;
  }
  names.forEach(name => {
    const chip = document.createElement("div");
    chip.className = "saved-chip";
    chip.innerHTML = `<span>${name}</span>`;

    const loadA = document.createElement("button");
    loadA.className = "secondary";
    loadA.textContent = "→ A";
    loadA.onclick = () => {
      teams.A = { name, roster: { ...saved[name] } };
      teamNameInput("A").value = name;
      persistCurrent();
      renderAll();
    };

    const loadB = document.createElement("button");
    loadB.className = "secondary";
    loadB.textContent = "→ B";
    loadB.onclick = () => {
      teams.B = { name, roster: { ...saved[name] } };
      teamNameInput("B").value = name;
      persistCurrent();
      renderAll();
    };

    const delBtn = document.createElement("button");
    delBtn.className = "danger";
    delBtn.textContent = "Delete";
    delBtn.onclick = () => {
      const t = getSavedTeams();
      delete t[name];
      setSavedTeams(t);
      renderSavedList();
    };

    chip.appendChild(loadA);
    chip.appendChild(loadB);
    chip.appendChild(delBtn);
    container.appendChild(chip);
  });
}

import { PLAYERS } from "./players.js";
import { teams, setProjections } from "./state.js";
import { fetchProjections } from "./projections.js";
import { loadCurrent, persistCurrent, getSavedTeams, setSavedTeams } from "./storage.js";
import { renderAll } from "./render.js";
import { renderSavedList } from "./savedTeams.js";
import { closeModal, renderModalList } from "./modal.js";

function teamNameInput(side) {
  return document.getElementById("teamName" + side);
}

loadCurrent();
teamNameInput("A").value = teams.A.name || "";
teamNameInput("B").value = teams.B.name || "";

renderAll();
renderSavedList();

document.getElementById("searchInput").addEventListener("input", (e) => renderModalList(e.target.value));
document.getElementById("closeModalBtn").addEventListener("click", closeModal);
document.getElementById("overlay").addEventListener("click", (e) => {
  if (e.target.id === "overlay") closeModal();
});

["A", "B"].forEach(side => {
  teamNameInput(side).addEventListener("input", () => {
    teams[side].name = teamNameInput(side).value;
    persistCurrent();
  });
  document.getElementById("saveBtn" + side).addEventListener("click", () => {
    const name = teamNameInput(side).value.trim();
    if (!name) { alert("Give this team a name first."); return; }
    const saved = getSavedTeams();
    saved[name] = { ...teams[side].roster };
    setSavedTeams(saved);
    persistCurrent();
    renderSavedList();
  });
  document.getElementById("clearBtn" + side).addEventListener("click", () => {
    if (!confirm("Clear this roster?")) return;
    teams[side].roster = {};
    persistCurrent();
    renderAll();
  });
});

fetchProjections(PLAYERS).then(map => {
  setProjections(map);
  renderAll();
});

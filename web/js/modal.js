import { PLAYERS } from "./players.js";
import { teams, activeTeam, activeSlot, setActive, clearActive } from "./state.js";
import { projFor, renderAll } from "./render.js";
import { persistCurrent } from "./storage.js";

export function openModal(side, slot) {
  setActive(side, slot);
  const label = teams[side].name || (side === "A" ? "Team A" : "Team B");
  document.getElementById("modalTitle").textContent = `Choose a ${slot.label} for ${label}`;
  document.getElementById("searchInput").value = "";
  document.getElementById("overlay").classList.add("open");
  renderModalList("");
  document.getElementById("searchInput").focus();
}

export function closeModal() {
  document.getElementById("overlay").classList.remove("open");
  clearActive();
}

export function renderModalList(query) {
  const list = document.getElementById("modalList");
  list.innerHTML = "";
  const q = query.trim().toLowerCase();
  // A player can only be on one side of the matchup at a time.
  const rosteredIds = new Set([
    ...Object.values(teams.A.roster),
    ...Object.values(teams.B.roster),
  ]);

  const candidates = PLAYERS.filter(p => activeSlot.eligible.includes(p.pos))
    .filter(p => !q || p.name.toLowerCase().includes(q) || p.team.toLowerCase().includes(q))
    .sort((a, b) => a.name.localeCompare(b.name));

  if (candidates.length === 0) {
    list.innerHTML = `<div class="empty-msg">No players match "${query}".</div>`;
    return;
  }

  candidates.forEach(p => {
    const takenElsewhere = rosteredIds.has(p.id) && teams[activeTeam].roster[activeSlot.key] !== p.id;
    const row = document.createElement("div");
    row.className = "player-row" + (takenElsewhere ? " taken" : "");
    row.innerHTML = `<div>
        <div class="name">${p.name}</div>
        <div class="meta">${p.pos} · ${p.team}${takenElsewhere ? " · already rostered" : ""}</div>
      </div>
      <div class="proj">${projFor(p.id).toFixed(1)}</div>`;
    if (!takenElsewhere) {
      row.onclick = () => {
        teams[activeTeam].roster[activeSlot.key] = p.id;
        persistCurrent();
        renderAll();
        closeModal();
      };
    }
    list.appendChild(row);
  });
}

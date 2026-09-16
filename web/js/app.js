import { PLAYERS, STARTER_SLOTS, BENCH_SLOTS, POS_COLOR, playerById } from "./players.js";
import { fetchProjections } from "./projections.js";
import { getSavedTeams, setSavedTeams, loadCurrentTeams, persistCurrentTeams } from "./storage.js";

let teams = loadCurrentTeams() || {
  A: { name: "", roster: {} },
  B: { name: "", roster: {} },
};
let activeTeam = null;
let activeSlot = null;
let projections = {}; // playerId -> projected points, filled in below

function teamNameInput(side) {
  return document.getElementById("teamName" + side);
}

function projFor(playerId) {
  return projections[playerId] ?? 0;
}

function teamStarterTotal(side) {
  return STARTER_SLOTS.reduce((sum, slot) => {
    const pid = teams[side].roster[slot.key];
    return sum + (pid !== undefined ? projFor(pid) : 0);
  }, 0);
}

function buildSlotCard(side, slot) {
  const pid = teams[side].roster[slot.key];
  const player = pid !== undefined ? playerById(pid) : null;
  const el = document.createElement("div");
  el.className = "slot";
  el.style.borderLeft = `4px solid ${POS_COLOR[slot.label] || "var(--border)"}`;
  el.onclick = () => openModal(side, slot);

  let inner = `<div class="slot-label">
      <span class="slot-badge" style="background:${POS_COLOR[slot.label] || "#888"}">${slot.label}</span>
    </div>`;
  if (player) {
    inner += `<div class="slot-player-name">${player.name}</div>
               <div class="slot-player-team">${player.pos} · ${player.team}</div>
               <div class="slot-proj">${projFor(player.id).toFixed(1)} proj</div>`;
  } else {
    inner += `<div class="slot-empty">+ Add ${slot.label}</div>`;
  }
  el.innerHTML = inner;

  if (player) {
    const x = document.createElement("span");
    x.className = "remove-x";
    x.textContent = "✕";
    x.onclick = (e) => {
      e.stopPropagation();
      delete teams[side].roster[slot.key];
      persistCurrentTeams(teams);
      renderAll();
    };
    el.appendChild(x);
  }
  return { el, proj: player ? projFor(player.id) : 0 };
}

function renderMatchup() {
  const container = document.getElementById("matchupRows");
  container.innerHTML = "";
  STARTER_SLOTS.forEach(slot => {
    const row = document.createElement("div");
    row.className = "matchup-row";

    const { el: cardA, proj: projA } = buildSlotCard("A", slot);
    const label = document.createElement("div");
    label.className = "matchup-slot-label";
    label.textContent = slot.label;
    const { el: cardB, proj: projB } = buildSlotCard("B", slot);

    if (projA !== projB) {
      (projA > projB ? cardA : cardB).classList.add("slot-win");
    }

    row.appendChild(cardA);
    row.appendChild(label);
    row.appendChild(cardB);
    container.appendChild(row);
  });
}

function renderBench() {
  ["A", "B"].forEach(side => {
    const container = document.getElementById("bench" + side);
    container.innerHTML = "";
    BENCH_SLOTS.forEach(slot => container.appendChild(buildSlotCard(side, slot).el));
  });
}

function renderScore() {
  const a = teamStarterTotal("A");
  const b = teamStarterTotal("B");
  const scoreA = document.getElementById("scoreA");
  const scoreB = document.getElementById("scoreB");
  scoreA.textContent = a.toFixed(1);
  scoreB.textContent = b.toFixed(1);
  scoreA.classList.toggle("winning", a > b);
  scoreB.classList.toggle("winning", b > a);
}

function renderAll() {
  renderMatchup();
  renderBench();
  renderScore();
}

function openModal(side, slot) {
  activeTeam = side;
  activeSlot = slot;
  const label = teams[side].name || (side === "A" ? "Team A" : "Team B");
  document.getElementById("modalTitle").textContent = `Choose a ${slot.label} for ${label}`;
  document.getElementById("searchInput").value = "";
  document.getElementById("overlay").classList.add("open");
  renderModalList("");
  document.getElementById("searchInput").focus();
}

function closeModal() {
  document.getElementById("overlay").classList.remove("open");
  activeTeam = null;
  activeSlot = null;
}

function renderModalList(query) {
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
        persistCurrentTeams(teams);
        renderAll();
        closeModal();
      };
    }
    list.appendChild(row);
  });
}

function renderSavedList() {
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
      persistCurrentTeams(teams);
      renderAll();
    };

    const loadB = document.createElement("button");
    loadB.className = "secondary";
    loadB.textContent = "→ B";
    loadB.onclick = () => {
      teams.B = { name, roster: { ...saved[name] } };
      teamNameInput("B").value = name;
      persistCurrentTeams(teams);
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
    persistCurrentTeams(teams);
  });
  document.getElementById("saveBtn" + side).addEventListener("click", () => {
    const name = teamNameInput(side).value.trim();
    if (!name) { alert("Give this team a name first."); return; }
    const saved = getSavedTeams();
    saved[name] = { ...teams[side].roster };
    setSavedTeams(saved);
    persistCurrentTeams(teams);
    renderSavedList();
  });
  document.getElementById("clearBtn" + side).addEventListener("click", () => {
    if (!confirm("Clear this roster?")) return;
    teams[side].roster = {};
    persistCurrentTeams(teams);
    renderAll();
  });
});

fetchProjections(PLAYERS).then(map => {
  projections = map;
  renderAll();
});

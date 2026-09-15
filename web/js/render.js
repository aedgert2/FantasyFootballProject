import { STARTER_SLOTS, BENCH_SLOTS, POS_COLOR, playerById } from "./players.js";
import { teams, projections } from "./state.js";
import { openModal } from "./modal.js";
import { persistCurrent } from "./storage.js";

export function projFor(playerId) {
  return projections[playerId] ?? 0;
}

export function teamStarterTotal(side) {
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
      persistCurrent();
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
    BENCH_SLOTS.forEach(slot => {
      const { el } = buildSlotCard(side, slot);
      container.appendChild(el);
    });
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

export function renderAll() {
  renderMatchup();
  renderBench();
  renderScore();
}

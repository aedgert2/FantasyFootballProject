import { teams } from "./state.js";

const STORAGE_KEY = "fantasyTeamBuilder.teams";
const CURRENT_KEY = "fantasyTeamBuilder.current";

export function getSavedTeams() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch (e) {
    return {};
  }
}

export function setSavedTeams(saved) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(saved)); } catch (e) { /* ignore */ }
}

export function loadCurrent() {
  try {
    const raw = localStorage.getItem(CURRENT_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed.teams) {
        teams.A = parsed.teams.A;
        teams.B = parsed.teams.B;
      }
    }
  } catch (e) { /* localStorage unavailable, start fresh */ }
}

export function persistCurrent() {
  try { localStorage.setItem(CURRENT_KEY, JSON.stringify({ teams })); } catch (e) { /* ignore */ }
}

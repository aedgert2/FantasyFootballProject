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

export function loadCurrentTeams() {
  try {
    const raw = localStorage.getItem(CURRENT_KEY);
    return raw ? JSON.parse(raw).teams : null;
  } catch (e) {
    return null;
  }
}

export function persistCurrentTeams(teams) {
  try { localStorage.setItem(CURRENT_KEY, JSON.stringify({ teams })); } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------------
// Shared mutable state for the two rosters and the player picker modal.
// `teams` is a stable object other modules import and mutate in place;
// `activeTeam`/`activeSlot`/`projections` are plain bindings, so they
// go through setters rather than being reassigned from other modules.
// ---------------------------------------------------------------------
export const teams = {
  A: { name: "", roster: {} },
  B: { name: "", roster: {} },
};

export let activeTeam = null;
export let activeSlot = null;
export let projections = {}; // playerId -> projected points

export function setActive(team, slot) {
  activeTeam = team;
  activeSlot = slot;
}

export function clearActive() {
  activeTeam = null;
  activeSlot = null;
}

export function setProjections(map) {
  projections = map;
}

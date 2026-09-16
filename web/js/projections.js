// ---------------------------------------------------------------------
// Projections. This is a deterministic mock so the matchup view has
// numbers to compare -- it is NOT a real forecast. When a real
// projections API exists, replace the body of fetchProjections() with
// the actual call (e.g. `return (await fetch(url)).json()`); nothing
// else in the app needs to change, since callers already treat this as
// async and keyed by player id.
// ---------------------------------------------------------------------
const RANGES = { QB: [14, 27], RB: [4, 22], WR: [3, 20], TE: [2, 14], K: [5, 11], DEF: [3, 12] };

function mockProjection(player) {
  const [lo, hi] = RANGES[player.pos];
  const frac = (player.id * 37) % 100 / 100;
  return +(lo + frac * (hi - lo)).toFixed(1);
}

export async function fetchProjections(players) {
  const map = {};
  players.forEach(p => { map[p.id] = mockProjection(p); });
  return map;
}

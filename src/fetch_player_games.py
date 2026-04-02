#!/usr/bin/env python3
"""
Fetch NUMS games for a Cartridge controller username and save them locally.

This resolves the controller address through Cartridge GraphQL, fetches owned
NUMS game IDs via Torii SQL, then decodes the packed game rows into readable
JSON for later analysis.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


CARTRIDGE_API_URL = "https://api.cartridge.gg/query"
TORII_SQL_URL = "https://api.cartridge.gg/x/nums-mainnet/torii/sql"
NUMS_COLLECTION_ADDRESS = "0x0282964e6c06a435fbf6ddf5a63bf4dc65d2ab879b30d320cf4d95543053aab5"
ZERO_ADDRESS = "0x" + "0" * 64
SLOT_COUNT = 18
SLOT_SIZE = 4096
PACKED_BASE = 16
POWER_NAMES = {
    0: "None",
    1: "Reroll",
    2: "High",
    3: "Low",
    4: "Swap",
    5: "DoubleUp",
    6: "Halve",
    7: "Mirror",
}
TRAP_NAMES = {
    0: "None",
    1: "Bomb",
    2: "Lucky",
    3: "Magnet",
    4: "UFO",
    5: "Windy",
}


def http_json(url: str, payload: bytes | None = None, headers: dict[str, str] | None = None) -> Any:
    request = Request(url, data=payload, headers=headers or {})
    try:
        with urlopen(request) as response:
            return json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def pad_address(address: str) -> str:
    return f"0x{address.lower().removeprefix('0x').rjust(64, '0')}"


def parse_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 16) if value.startswith("0x") else int(value)
    return 0


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return parse_int(value) != 0


def unpack_base(value: int, count: int, base: int) -> list[int]:
    items = []
    current = value
    for _ in range(count):
        items.append(current % base)
        current //= base
    return items


def unpack_slots(packed: Any, slot_count: int) -> list[int]:
    return unpack_base(parse_int(packed), slot_count, SLOT_SIZE)


def unpack_powers(packed: Any) -> list[int]:
    value = parse_int(packed)
    powers = []
    while value > 0:
        power = value % PACKED_BASE
        if power != 0:
            powers.append(power)
        value //= PACKED_BASE
    return powers


def unpack_bitmap(packed: Any, count: int) -> list[bool]:
    value = parse_int(packed)
    return [((value >> idx) & 1) == 1 for idx in range(count)]


def resolve_controller_address(username: str) -> str:
    payload = json.dumps(
        {
            "query": (
                "query AddressByUsername($username: String!) { "
                "account(username: $username) { "
                "username controllers(first: 1) { edges { node { address } } } "
                "} }"
            ),
            "variables": {"username": username},
        }
    ).encode("utf-8")
    response = http_json(
        CARTRIDGE_API_URL,
        payload=payload,
        headers={"content-type": "application/json"},
    )
    edges = (
        response.get("data", {})
        .get("account", {})
        .get("controllers", {})
        .get("edges", [])
    )
    if not edges:
        raise RuntimeError(f"No controller address found for username {username!r}")
    return edges[0]["node"]["address"]


def torii_sql(query: str) -> list[dict[str, Any]]:
    encoded = quote(query, safe="")
    data = http_json(f"{TORII_SQL_URL}?query={encoded}")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected Torii response: {data!r}")
    return data


def fetch_game_ids(controller_address: str) -> list[int]:
    query = (
        "SELECT token_id FROM token_transfers "
        f"WHERE contract_address = '{NUMS_COLLECTION_ADDRESS}' "
        f"AND to_address = '{pad_address(controller_address)}' "
        f"AND from_address = '{ZERO_ADDRESS}'"
    )
    rows = torii_sql(query)
    ids = []
    for row in rows:
        token_id = row.get("token_id", "")
        game_hex = token_id.split(":")[-1]
        ids.append(parse_int(game_hex))
    return sorted(set(ids))


def fetch_game_rows(game_ids: list[int]) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, len(game_ids), 25):
        chunk = game_ids[start:start + 25]
        ids = ", ".join(f"'0x{game_id:016x}'" for game_id in chunk)
        query = f'SELECT * FROM "NUMS-Game" WHERE id IN ({ids}) ORDER BY id'
        rows.extend(torii_sql(query))
    return rows


def decode_game(row: dict[str, Any]) -> dict[str, Any]:
    slot_count = parse_int(row.get("slot_count")) or SLOT_COUNT
    selected_powers = unpack_powers(row.get("selected_powers"))
    selectable_powers = unpack_powers(row.get("selectable_powers"))
    enabled_powers = unpack_bitmap(row.get("enabled_powers"), 3)
    disabled_traps = unpack_bitmap(row.get("disabled_traps"), slot_count)
    traps = unpack_base(parse_int(row.get("traps")), slot_count, PACKED_BASE)
    slots = unpack_slots(row.get("slots"), slot_count)

    active_traps = []
    for idx, trap in enumerate(traps):
        if trap != 0 and not disabled_traps[idx]:
            active_traps.append({"slot": idx, "trap": TRAP_NAMES.get(trap, f"Trap{trap}")})

    over_ts = parse_int(row.get("over"))
    return {
        "id": parse_int(row.get("id")),
        "level": parse_int(row.get("level")),
        "slot_count": slot_count,
        "number": parse_int(row.get("number")),
        "next_number": parse_int(row.get("next_number")),
        "claimed": parse_bool(row.get("claimed")),
        "over": over_ts != 0,
        "over_timestamp": over_ts or None,
        "over_at": datetime.fromtimestamp(over_ts, tz=timezone.utc).isoformat() if over_ts else None,
        "reward": parse_int(row.get("reward")),
        "slots": slots,
        "filled_slots": sum(1 for slot in slots if slot != 0),
        "selectable_powers": [POWER_NAMES.get(power, str(power)) for power in selectable_powers],
        "selected_powers": [POWER_NAMES.get(power, str(power)) for power in selected_powers],
        "enabled_power_indices": [idx for idx, enabled in enumerate(enabled_powers) if enabled],
        "traps": [TRAP_NAMES.get(trap, str(trap)) for trap in traps],
        "disabled_traps": disabled_traps,
        "active_traps": active_traps,
    }


def build_output(username: str, controller_address: str, games: list[dict[str, Any]]) -> dict[str, Any]:
    levels = [game["level"] for game in games]
    return {
        "username": username,
        "controller_address": controller_address,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "games": len(games),
            "avg_level": round(sum(levels) / len(levels), 2) if levels else 0.0,
            "best_level": max(levels) if levels else 0,
            "worst_level": min(levels) if levels else 0,
        },
        "games": games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NUMS games for a controller username")
    parser.add_argument("--username", required=True, help="Controller username, e.g. krump")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/player_games/krump.json"),
        help="Where to save the decoded JSON output",
    )
    args = parser.parse_args()

    controller_address = resolve_controller_address(args.username)
    game_ids = fetch_game_ids(controller_address)
    games = [decode_game(row) for row in fetch_game_rows(game_ids)]
    payload = build_output(args.username, controller_address, games)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    summary = payload["summary"]
    print(f"Saved {summary['games']} games for {args.username} -> {args.output}")
    print(f"Controller: {controller_address}")
    print(f"Average level: {summary['avg_level']:.2f} | Best: {summary['best_level']} | Worst: {summary['worst_level']}")


if __name__ == "__main__":
    main()

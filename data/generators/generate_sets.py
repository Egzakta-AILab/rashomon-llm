#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate the built-in rashomon-llm prompt sets (agent40, math40).

Deterministic: one seed, fixed templates, no LLM involvement. Every set ships
with a manifest (generator, version, seed, date, per-item sha256, license,
provenance statements). Re-running this script must reproduce byte-identical
sets except for the manifest's generated_at field.

    python data/generators/generate_sets.py            # writes src/rashomon_llm/data/sets/
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

GENERATOR = {"name": "rashomon-llm generate_sets.py", "version": "1.0.0"}
SEED = 20261028
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src", "rashomon_llm", "data", "sets")

LICENSE = "Apache-2.0"
PROVENANCE = {
    "not_derived_from_any_benchmark": True,
    "llm_involvement": "none — templates written by a human author, instances "
                       "filled by this deterministic script",
    "license": LICENSE,
}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


def _str_prop(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _int_prop(desc: str) -> dict:
    return {"type": "integer", "description": desc}


# ---------------------------------------------------------------------------
# agent40: five families, eight instances each; outcome = first tool call

def _agent_items(rng: random.Random) -> list[dict]:
    items: list[dict] = []

    tools_inventory = [
        _tool("check_inventory", "Check the stock level of a product",
              {"sku": _str_prop("product SKU")}, ["sku"]),
        _tool("get_unit_price", "Get the unit price of a product in EUR",
              {"sku": _str_prop("product SKU")}, ["sku"]),
        _tool("place_order", "Place a purchase order",
              {"sku": _str_prop("product SKU"), "quantity": _int_prop("units")},
              ["sku", "quantity"]),
    ]
    for i in range(8):
        sku = f"SKU-{1000 + i * 7}"
        qty = 2 + rng.randrange(6)
        items.append({
            "id": f"agent-inventory-{i:02d}", "family": "agent_inventory",
            "messages": [{"role": "user", "content": (
                f"A customer wants to buy {qty} units of product {sku}. "
                f"First check the inventory for {sku}, then report the total price. "
                f"Start by calling the inventory check tool.")}],
            "tools": tools_inventory,
        })

    tools_weather = [
        _tool("get_weather", "Get the current weather for a city",
              {"city": _str_prop("city name")}, ["city"]),
        _tool("convert_units", "Convert a temperature between Celsius and Fahrenheit",
              {"value": _int_prop("numeric value"), "from_unit": _str_prop("'C' or 'F'"),
               "to": _str_prop("'C' or 'F'")}, ["value", "from_unit", "to"]),
    ]
    cities = ["Belgrade", "Vienna", "Zurich", "Oslo", "Lisbon", "Tallinn", "Porto", "Graz"]
    for i, city in enumerate(cities):
        items.append({
            "id": f"agent-weather-{i:02d}", "family": "agent_weather",
            "messages": [{"role": "user", "content": (
                f"What is the current temperature in {city}, converted to Fahrenheit? "
                f"Get the weather for {city} first.")}],
            "tools": tools_weather,
        })

    tools_booking = [
        _tool("find_reservation", "Find a reservation by guest email",
              {"email": _str_prop("guest email")}, ["email"]),
        _tool("cancel_reservation", "Cancel a reservation",
              {"reservation_id": _str_prop("reservation id")}, ["reservation_id"]),
        _tool("list_slots", "List available table slots for a date",
              {"date": _str_prop("YYYY-MM-DD")}, ["date"]),
    ]
    for i in range(8):
        email = f"guest{i}@example.org"
        items.append({
            "id": f"agent-booking-{i:02d}", "family": "agent_booking",
            "messages": [{"role": "user", "content": (
                f"The guest {email} wants to move their dinner reservation to a free slot on "
                f"2026-11-{10 + i:02d}. Find their reservation first, before doing anything else.")}],
            "tools": tools_booking,
        })

    tools_directory = [
        _tool("find_employee", "Find an employee by name",
              {"name": _str_prop("full name")}, ["name"]),
        _tool("get_extension", "Get an employee's phone extension",
              {"employee_id": _str_prop("employee id")}, ["employee_id"]),
        _tool("send_message", "Send an internal message",
              {"employee_id": _str_prop("employee id"), "text": _str_prop("message text")},
              ["employee_id", "text"]),
    ]
    names = ["Ana Jovanovic", "Marko Petrovic", "Jelena Nikolic", "Stefan Djordjevic",
             "Milica Stojanovic", "Nikola Ilic", "Jovana Pavlovic", "Luka Simic"]
    for i, name in enumerate(names):
        items.append({
            "id": f"agent-directory-{i:02d}", "family": "agent_directory",
            "messages": [{"role": "user", "content": (
                f"Send a short welcome message to our new colleague {name}. "
                f"You will need their employee id — look {name} up in the directory first.")}],
            "tools": tools_directory,
        })

    tools_shipping = [
        _tool("get_package_weight", "Get a parcel's weight in kg",
              {"parcel_id": _str_prop("parcel id")}, ["parcel_id"]),
        _tool("quote_shipping", "Quote the shipping price for a parcel",
              {"parcel_id": _str_prop("parcel id"), "destination": _str_prop("destination city")},
              ["parcel_id", "destination"]),
        _tool("schedule_pickup", "Schedule a parcel pickup",
              {"parcel_id": _str_prop("parcel id"), "date": _str_prop("YYYY-MM-DD")},
              ["parcel_id", "date"]),
    ]
    for i in range(8):
        parcel = f"PCL-{5000 + i * 13}"
        items.append({
            "id": f"agent-shipping-{i:02d}", "family": "agent_shipping",
            "messages": [{"role": "user", "content": (
                f"We need to ship parcel {parcel} to Split. Give me a shipping quote. "
                f"Start by checking the weight of {parcel}.")}],
            "tools": tools_shipping,
        })

    return items


# ---------------------------------------------------------------------------
# math40: five templates, eight instances each; outcome = final number

def _math_items(rng: random.Random) -> list[dict]:
    items: list[dict] = []

    for i in range(8):
        a = 2 + rng.randrange(9)
        b = rng.randrange(1, 50)
        x = rng.randrange(-10, 15)
        c = a * x + b
        items.append({
            "id": f"math-linear-{i:02d}", "family": "math_linear",
            "messages": [{"role": "user", "content": (
                f"Solve for x: {a}x + {b} = {c}. Answer with just the number.")}],
        })

    for i in range(8):
        p = 5 * (1 + rng.randrange(19))
        v = 40 + 20 * rng.randrange(50)
        items.append({
            "id": f"math-percent-{i:02d}", "family": "math_percent",
            "messages": [{"role": "user", "content": (
                f"What is {p}% of {v}? Answer with just the number.")}],
        })

    for i in range(8):
        h = 2 + rng.randrange(6)
        speed = 40 + 10 * rng.randrange(12)
        d = speed * h
        items.append({
            "id": f"math-speed-{i:02d}", "family": "math_speed",
            "messages": [{"role": "user", "content": (
                f"A car travels {d} km in {h} hours at a constant speed. "
                f"What is its average speed in km/h? Answer with just the number.")}],
        })

    for i in range(8):
        p = 500 + 100 * rng.randrange(40)
        r = 2 + rng.randrange(8)
        y = 1 + rng.randrange(10)
        items.append({
            "id": f"math-interest-{i:02d}", "family": "math_interest",
            "messages": [{"role": "user", "content": (
                f"An investment of {p} EUR earns {r}% simple interest per year. "
                f"What is its total value after {y} years? Answer with just the number.")}],
        })

    for i in range(8):
        length = 3 + rng.randrange(20)
        w = 2 + rng.randrange(15)
        items.append({
            "id": f"math-area-{i:02d}", "family": "math_area",
            "messages": [{"role": "user", "content": (
                f"A rectangle has length {length} m and width {w} m. "
                f"What is its area in square meters? Answer with just the number.")}],
        })

    return items


def _item_sha(item: dict) -> str:
    body = json.dumps({"id": item["id"], "messages": item["messages"],
                       "tools": item.get("tools")}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _write_set(name: str, items: list[dict]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    manifest = {
        "name": name,
        "generator": GENERATOR,
        "seed": SEED,
        "generated_at": time.strftime("%Y-%m-%d"),
        "n_items": len(items),
        "families": sorted({it["family"] for it in items}),
        "items_sha256": {it["id"]: _item_sha(it) for it in items},
        **PROVENANCE,
    }
    with open(os.path.join(OUT_DIR, f"{name}.manifest.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{name}: {len(items)} items -> {path}")


def main() -> None:
    rng = random.Random(SEED)
    _write_set("agent40", _agent_items(random.Random(SEED)))
    _write_set("math40", _math_items(rng))


if __name__ == "__main__":
    main()

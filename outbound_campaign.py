"""Run an outbound calling campaign.

Originates outbound calls to a list of contacts through an Asterisk SIP trunk
(via ARI), presenting one of your DIDs as caller ID. The answered leg is
bridged to VoiceOS by serve_telephony.py, so START THAT FIRST — this script
only triggers the calls.

    # 1) terminal A: media bridge that answers the calls
    python serve_telephony.py

    # 2) terminal B: originate the calls (ARI creds via env)
    export ARI_BASE_URL=http://asterisk:8088 ARI_USER=ari ARI_PASSWORD=secret
    python outbound_campaign.py contacts.json \
        --trunk telnyx --caller-id +15559876543 --max-concurrency 20 --delay 0.5

contacts.json is a list of objects:
    [{"number": "+15551234567", "name": "Asha", "consented": true}, ...]

⚠️  Consent gate is ON by default: contacts without "consented": true are
skipped (TCPA — AI voices need prior express consent). Pass --no-consent-check
ONLY if you have a lawful basis; you own that decision.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from voiceos.telephony.campaign import CampaignRunner, Contact, make_ari_originator
from voiceos.utils.logging import setup_logging


def load_contacts(path: str) -> list[Contact]:
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    return [
        Contact(
            number=row["number"],
            consented=bool(row.get("consented", False)),
            name=row.get("name", ""),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


async def run(args) -> None:
    base_url = os.environ.get("ARI_BASE_URL")
    user = os.environ.get("ARI_USER")
    password = os.environ.get("ARI_PASSWORD")
    if not (base_url and user and password):
        sys.exit("set ARI_BASE_URL, ARI_USER, ARI_PASSWORD in the environment")

    contacts = load_contacts(args.contacts)
    originate = make_ari_originator(
        trunk=args.trunk, context=args.context,
        base_url=base_url, username=user, password=password,
    )
    runner = CampaignRunner(
        originate, caller_id=args.caller_id,
        max_concurrency=args.max_concurrency, call_delay=args.delay,
        require_consent=not args.no_consent_check,
        on_result=lambda r: print(f"  {r.status:20} {r.contact.number}"),
    )

    print(f"Dialing {len(contacts)} contacts through '{args.trunk}' as {args.caller_id}\n")
    results = await runner.run(contacts)

    tally: dict[str, int] = {}
    for r in results:
        tally[r.status] = tally.get(r.status, 0) + 1
    print("\nSummary: " + ", ".join(f"{n} {s}" for s, n in sorted(tally.items())))


def main() -> None:
    parser = argparse.ArgumentParser(prog="voiceos-campaign")
    parser.add_argument("contacts", help="path to contacts JSON")
    parser.add_argument("--trunk", required=True, help="Asterisk trunk name, e.g. telnyx")
    parser.add_argument("--caller-id", required=True, help="a DID you own, E.164")
    parser.add_argument("--context", default="voiceos-outbound", help="dialplan context")
    parser.add_argument("--max-concurrency", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between originations")
    parser.add_argument(
        "--no-consent-check", action="store_true",
        help="dial contacts even without consented:true (you own the legal basis)",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nCampaign interrupted.")


if __name__ == "__main__":
    main()

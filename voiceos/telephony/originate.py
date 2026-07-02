"""Outbound call origination.

Trigger an outbound call from the media server and let it connect back to
the VoiceOS AudioSocket/media bridge. Two paths:

Asterisk (ARI) — implemented here via httpx: originate a channel to the SIP
trunk, with the caller ID set to the DID you're calling *from*, landing in a
dialplan extension that runs AudioSocket() so the call's audio bridges to
VoiceOS.

    await ari_originate(
        endpoint="PJSIP/+15551234567@telnyx",   # who to call, via the trunk
        caller_id="+15559876543",               # the DID you own -> caller ID
        context="voiceos-outbound", extension="s",
        base_url="http://asterisk:8088", username="ari", password="secret",
    )

FreeSWITCH (ESL) — no Python ESL client is bundled; the equivalent command is:

    originate {origination_caller_id_number=+15559876543,\
               ignore_early_media=true}sofia/gateway/telnyx/+15551234567 \
               &socket('voiceos-host:8090 async full')     # mod_audio_socket
    # or  ... &lua(bridge_to_ai.lua)  /  park + mod_audio_stream to a WebSocket

Both set the outbound caller ID to a specific DID and route the answered
call into the audio bridge, which spawns a VoiceOS session for that call.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


async def ari_originate(
    endpoint: str,
    *,
    caller_id: str,
    context: str,
    extension: str = "s",
    priority: int = 1,
    base_url: str,
    username: str,
    password: str,
    variables: dict | None = None,
    timeout: float = 30.0,
    client: httpx.AsyncClient | None = None,
) -> dict:
    """Originate an outbound Asterisk call via ARI. Returns the channel JSON.

    `endpoint` is who to dial through the trunk (e.g. PJSIP/<e164>@<trunk>);
    `caller_id` is the DID to present (your owned number); `context`/`extension`
    is the dialplan entry that runs AudioSocket() to bridge audio to VoiceOS.
    """
    params = {
        "endpoint": endpoint,
        "callerId": caller_id,
        "context": context,
        "extension": extension,
        "priority": priority,
    }
    payload = {"variables": variables or {}}

    owns_client = client is None
    if client is None:
        client = httpx.AsyncClient(
            base_url=base_url, auth=(username, password),
            timeout=httpx.Timeout(timeout, connect=5.0),
        )
    try:
        response = await client.post("/ari/channels", params=params, json=payload)
        response.raise_for_status()
        channel = response.json()
        logger.info("originated outbound channel %s to %s", channel.get("id"), endpoint)
        return channel
    finally:
        if owns_client:
            await client.aclose()

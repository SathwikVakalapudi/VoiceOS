"""Outbound ARI origination test (mocked HTTP — no live Asterisk)."""

import httpx

from voiceos.telephony.originate import ari_originate


async def test_ari_originate_builds_correct_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"id": "chan-1", "state": "Down"})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://asterisk:8088",
        auth=("ari", "secret"),
    )
    channel = await ari_originate(
        endpoint="PJSIP/+15551234567@telnyx",
        caller_id="+15559876543",
        context="voiceos-outbound",
        base_url="http://asterisk:8088",
        username="ari",
        password="secret",
        client=client,
    )
    await client.aclose()

    assert channel["id"] == "chan-1"
    assert seen["method"] == "POST"
    assert seen["path"] == "/ari/channels"
    assert seen["params"]["endpoint"] == "PJSIP/+15551234567@telnyx"
    assert seen["params"]["callerId"] == "+15559876543"    # DID as caller ID
    assert seen["params"]["context"] == "voiceos-outbound"
    assert seen["auth"] is not None                        # basic auth sent

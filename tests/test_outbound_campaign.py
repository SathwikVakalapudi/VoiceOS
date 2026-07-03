"""Outbound campaign runner tests: consent gate, concurrency, failures."""

import asyncio

import httpx

from voiceos.telephony.campaign import (
    CampaignRunner,
    Contact,
    make_ari_originator,
)


async def test_consent_gate_skips_non_consented_contacts():
    dialed = []

    async def originate(number, caller_id):
        dialed.append(number)
        return {"id": "chan-" + number}

    runner = CampaignRunner(originate, caller_id="+15559990000")
    results = await runner.run(
        [Contact("+1111", consented=True), Contact("+2222", consented=False)]
    )

    assert dialed == ["+1111"]                                  # only consented dialed
    by_num = {r.contact.number: r for r in results}
    assert by_num["+1111"].status == "originated" and by_num["+1111"].ok
    assert by_num["+2222"].status == "skipped_no_consent"


async def test_require_consent_false_dials_everyone():
    dialed = []

    async def originate(number, caller_id):
        dialed.append(number)
        return {"id": number}

    runner = CampaignRunner(originate, caller_id="+1", require_consent=False)
    await runner.run([Contact("+a"), Contact("+b")])
    assert sorted(dialed) == ["+a", "+b"]


async def test_a_single_failure_does_not_abort_the_campaign():
    async def originate(number, caller_id):
        if number == "+boom":
            raise RuntimeError("trunk rejected")
        return {"id": number}

    runner = CampaignRunner(
        originate, caller_id="+1", require_consent=False
    )
    results = await runner.run([Contact("+ok"), Contact("+boom"), Contact("+ok2")])
    status = {r.contact.number: r.status for r in results}
    assert status == {"+ok": "originated", "+boom": "failed", "+ok2": "originated"}
    assert next(r for r in results if r.contact.number == "+boom").error == "trunk rejected"


async def test_concurrency_is_bounded():
    active = 0
    peak = 0

    async def originate(number, caller_id):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"id": number}

    runner = CampaignRunner(
        originate, caller_id="+1", require_consent=False, max_concurrency=2
    )
    await runner.run([Contact(str(i)) for i in range(10)])
    assert peak <= 2


async def test_on_result_callback_fires_per_contact():
    seen = []

    async def originate(number, caller_id):
        return {"id": number}

    runner = CampaignRunner(
        originate, caller_id="+1", require_consent=False, on_result=seen.append
    )
    await runner.run([Contact("+x"), Contact("+y")])
    assert {r.contact.number for r in seen} == {"+x", "+y"}


async def test_dry_run_places_no_calls_but_still_applies_consent_gate():
    dialed = []

    async def originate(number, caller_id):
        dialed.append(number)                                  # must never happen
        return {"id": number}

    runner = CampaignRunner(originate, caller_id="+1", dry_run=True)
    results = await runner.run(
        [Contact("+ok", consented=True), Contact("+nc", consented=False)]
    )

    assert dialed == []                                        # nothing originated
    status = {r.contact.number: r.status for r in results}
    assert status == {"+ok": "dry_run", "+nc": "skipped_no_consent"}
    assert not any(r.ok for r in results)                      # dry_run is not "originated"


async def test_dry_run_with_consent_check_off_previews_everyone():
    async def originate(number, caller_id):
        raise AssertionError("should not be called in dry run")

    runner = CampaignRunner(
        originate, caller_id="+1", dry_run=True, require_consent=False
    )
    results = await runner.run([Contact("+a"), Contact("+b")])
    assert all(r.status == "dry_run" for r in results)


async def test_ari_originator_builds_pjsip_endpoint_through_trunk():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json={"id": "chan-1"})

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://asterisk:8088",
        auth=("ari", "secret"),
    )
    originate = make_ari_originator(
        trunk="telnyx", base_url="http://asterisk:8088",
        username="ari", password="secret", client=client,
    )
    channel = await originate("+15551234567", "+15559876543")
    await client.aclose()

    assert channel["id"] == "chan-1"
    assert seen["params"]["endpoint"] == "PJSIP/+15551234567@telnyx"
    assert seen["params"]["callerId"] == "+15559876543"
    assert seen["params"]["context"] == "voiceos-outbound"

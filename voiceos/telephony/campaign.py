"""Outbound calling campaign runner.

Drives a list of contacts into outbound calls at a controlled rate, with two
production guardrails wired in by default:

  * **Consent gate (TCPA/FCC).** AI voices are "artificial" under the TCPA;
    US outbound needs prior express consent. Contacts without `consented=True`
    are skipped with a logged reason instead of dialed — you cannot forget it.
  * **Bounded concurrency + pacing.** A semaphore caps simultaneous calls
    (protect the media servers and GPU worker pool) and an optional inter-call
    delay throttles origination to stay within trunk/CPS limits and avoid
    tripping carrier spam heuristics.

The actual origination is injected (`originate` coroutine) so the runner is
testable without a live media server; `make_ari_originator()` builds one from
`ari_originate` for the Asterisk/ARI path.

Set `dry_run=True` to preview a run: the consent gate still applies, so you
see exactly who would be dialed (status "dry_run") versus skipped, but no call
is placed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from voiceos.telephony.originate import ari_originate

logger = logging.getLogger(__name__)

# originate(number, caller_id) -> channel/job info (provider-specific dict).
Originator = Callable[[str, str], Awaitable[dict]]


@dataclass
class Contact:
    number: str                       # E.164, e.g. "+15551234567"
    consented: bool = False           # prior express consent on file?
    name: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class CallResult:
    contact: Contact
    status: str                       # "originated" | "skipped_no_consent" | "failed"
    channel: dict | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "originated"


class CampaignRunner:
    def __init__(
        self,
        originate: Originator,
        *,
        caller_id: str,
        max_concurrency: int = 20,
        call_delay: float = 0.0,
        require_consent: bool = True,
        dry_run: bool = False,
        on_result: Callable[[CallResult], None] | None = None,
    ) -> None:
        self._originate = originate
        self._caller_id = caller_id
        self._call_delay = call_delay
        self._require_consent = require_consent
        self._dry_run = dry_run
        self._on_result = on_result
        self._sem = asyncio.Semaphore(max_concurrency)

    async def _place(self, contact: Contact) -> CallResult:
        if self._require_consent and not contact.consented:
            logger.warning("skipping %s: no prior express consent on file", contact.number)
            result = CallResult(contact, "skipped_no_consent")
        elif self._dry_run:
            # Consent-passed and would be dialed — but place no call.
            logger.info("[dry-run] would call %s as %s", contact.number, self._caller_id)
            result = CallResult(contact, "dry_run")
        else:
            async with self._sem:
                try:
                    channel = await self._originate(contact.number, self._caller_id)
                    logger.info("originated call to %s", contact.number)
                    result = CallResult(contact, "originated", channel=channel)
                except Exception as exc:  # keep the campaign going on a single failure
                    logger.exception("failed to originate call to %s", contact.number)
                    result = CallResult(contact, "failed", error=str(exc))
        if self._on_result is not None:
            self._on_result(result)
        return result

    async def run(self, contacts: list[Contact]) -> list[CallResult]:
        """Dial every contact, honoring concurrency, pacing, and consent.

        Returns one CallResult per contact, in input order.
        """
        tasks = []
        for contact in contacts:
            tasks.append(asyncio.create_task(self._place(contact)))
            if self._call_delay and not self._dry_run and contact is not contacts[-1]:
                await asyncio.sleep(self._call_delay)  # pace origination
        return await asyncio.gather(*tasks)


def make_ari_originator(
    *,
    trunk: str,
    context: str = "voiceos-outbound",
    extension: str = "s",
    base_url: str,
    username: str,
    password: str,
    client=None,
) -> Originator:
    """Build an `Originator` that dials via Asterisk ARI through `trunk`.

    Each call becomes `PJSIP/<number>@<trunk>` presenting `caller_id`, landing
    in the dialplan `context`/`extension` that runs AudioSocket() to bridge the
    answered call's audio to VoiceOS.
    """

    async def originate(number: str, caller_id: str) -> dict:
        return await ari_originate(
            endpoint=f"PJSIP/{number}@{trunk}",
            caller_id=caller_id,
            context=context,
            extension=extension,
            base_url=base_url,
            username=username,
            password=password,
            client=client,
        )

    return originate

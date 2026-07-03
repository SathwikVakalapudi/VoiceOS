# Telephony: connecting VoiceOS to a SIP trunk (inbound + outbound, at scale)

This is the production setup for putting VoiceOS on real phone numbers using
a **provider SIP trunk + DIDs** (Telnyx/Plivo/Signalwire), a **media server**
(Asterisk or FreeSWITCH) that terminates SIP/RTP, and a **socket audio
bridge** to VoiceOS. VoiceOS never speaks SIP or RTP — the media server does,
and hands each call's audio to VoiceOS over a socket via the
`AudioTransport` seam.

```
Caller <-> PSTN/cellular <-> [SIP trunk: Telnyx/Plivo]
                                   | SIP + RTP (G.711 8k)
                             [Asterisk / FreeSWITCH]   <- media plane (cheap CPU)
                                   | AudioSocket (TCP) / mod_audio_stream (WS)
                             [VoiceOS session per call] <- AI plane (GPU bound)
                                   |
                             VAD -> STT -> LLM -> TTS
```

> ⚠️ **Not runnable from this repo alone.** You need a real SIP trunk, a
> media server host, and a public IP. The VoiceOS-side code
> (`voiceos/telephony/`) is implemented and unit-tested; the media-server
> config below is verified against vendor docs but must be deployed by you.
> The exact FreeSWITCH `mod_audio_stream` JSON envelope should be confirmed
> against your module version.

---

## 1. Which bridge?

| Bridge | Media server | Transport | VoiceOS status |
|---|---|---|---|
| **AudioSocket** (recommended) | Asterisk | TCP, linear 8k PCM, trivial framing | **Implemented + tested** (`voiceos/telephony/audiosocket.py`) |
| **mod_audio_stream** | FreeSWITCH | WebSocket, mu-law 8k | **Implemented + tested** (`voiceos/telephony/websocket.py`, `--protocol binary`) |
| **Twilio Media Streams** | Twilio | WebSocket, JSON/base64 mu-law | **Implemented + tested** (`voiceos/telephony/websocket.py`, `--protocol twilio`) |
| **ARI externalMedia** | Asterisk | RTP to your app | Heaviest; only if you want raw RTP |

AudioSocket is the simplest robust bidirectional bridge and needs no extra
Python deps (stdlib asyncio TCP). Use it unless you're committed to FreeSWITCH.

---

## 2. Asterisk (AudioSocket) — the implemented path

### 2a. SIP trunk (`pjsip.conf`) — Telnyx example, IP-authenticated
```ini
[telnyx-transport]
type=transport
protocol=udp
bind=0.0.0.0:5060

[telnyx]
type=endpoint
context=voiceos-inbound
disallow=all
allow=ulaw,alaw            ; G.711; add opus if your trunk negotiates it
direct_media=no            ; keep media on the server so we can bridge it
dtmf_mode=rfc4733          ; out-of-band DTMF (RFC 2833/4733)
aors=telnyx
outbound_auth=telnyx-auth  ; omit if the trunk uses pure IP auth

[telnyx]
type=identify              ; inbound: trust the trunk's signaling IPs
endpoint=telnyx
match=192.76.120.0/24      ; <- Telnyx signaling ranges (check current docs)

[telnyx]
type=aor
contact=sip:sip.telnyx.com

[telnyx-auth]
type=auth
auth_type=userpass
username=YOUR_TRUNK_USER
password=YOUR_TRUNK_SECRET
```

### 2b. Dialplan (`extensions.conf`) — inbound DID + outbound
```ini
[voiceos-inbound]                       ; incoming calls to your DID
exten => _X.,1,Answer()
 same => n,AudioSocket(${UUID()},voiceos-host:8090)   ; bridge audio to VoiceOS
 same => n,Hangup()

[voiceos-outbound]                      ; used by ARI-originated outbound calls
exten => s,1,Answer()
 same => n,AudioSocket(${UUID()},voiceos-host:8090)
 same => n,Hangup()
```

### 2c. Outbound origination (from a specific DID)
VoiceOS triggers the call via ARI (`voiceos/telephony/originate.py`):
```python
await ari_originate(
    endpoint="PJSIP/+15551234567@telnyx",  # callee, through the trunk
    caller_id="+15559876543",              # your DID -> outbound caller ID
    context="voiceos-outbound", extension="s",
    base_url="http://asterisk:8088", username="ari", password="secret",
)
```
Asterisk dials out; when answered, the dialplan runs `AudioSocket(...)`, which
connects to the VoiceOS `AudioSocketServer` and spawns a session for that call.

### 2d. Run the VoiceOS side
```python
import asyncio
from voiceos.config.settings import get_settings
from voiceos.pipeline.pipeline import VoicePipeline
from voiceos.telephony.audiosocket import AudioSocketServer

settings = get_settings()
def make_session(transport):            # one pipeline per call, bound to the call's audio
    return VoicePipeline(settings, transport=transport)

server = AudioSocketServer(
    make_session, host="0.0.0.0", port=8090,
    input_sample_rate=settings.audio.input_sample_rate,
    frame_size=settings.audio.frame_size,
)
asyncio.run(server.serve_forever())
```

---

## 3. FreeSWITCH (mod_audio_stream) — the WebSocket path

### 3a. SIP trunk gateway (`sofia` external profile)
```xml
<gateway name="telnyx">
  <param name="proxy" value="sip.telnyx.com"/>
  <param name="register" value="false"/>       <!-- IP-authenticated trunk -->
  <param name="username" value="not-used"/>
  <param name="password" value="not-used"/>
  <param name="caller-id-in-from" value="true"/>
</gateway>
```

### 3b. Inbound dialplan — stream audio to a VoiceOS WebSocket
```xml
<extension name="voiceos-inbound">
  <condition field="destination_number" expression="^(\+?\d+)$">
    <action application="answer"/>
    <!-- streams mu-law 8k frames over WS; confirm your module's arg format -->
    <action application="audio_stream"
            data="wss://voiceos-host:8091/ws start mono 8k"/>
  </condition>
</extension>
```

### 3c. Outbound origination (ESL)
```
originate {origination_caller_id_number=+15559876543,ignore_early_media=true}\
  sofia/gateway/telnyx/+15551234567 &socket('voiceos-host:8090 async full')
```

On the VoiceOS side, run the WebSocket bridge — it builds a
`MediaStreamTransport` + `VoicePipeline` per connection, decodes inbound audio,
and streams TTS back, including a barge-in "clear" flush:

```bash
pip install websockets                        # optional telephony dep
python serve_telephony.py --bridge websocket --protocol binary --port 8091
#   --protocol binary  -> FreeSWITCH mod_audio_stream (raw mu-law frames)
#   --protocol twilio  -> Twilio Media Streams (JSON/base64 mu-law envelope)
```

Confirm your `mod_audio_stream` build's exact frame format; the wire framing
is isolated in `MediaProtocol` (`voiceos/telephony/websocket.py`) so only that
small class needs adjusting if your module differs.

## 3d. Outbound campaigns (both bridges)

`outbound_campaign.py` originates a list of contacts through the trunk with a
**TCPA consent gate** (skips contacts without `consented: true`), bounded
concurrency, and pacing. Start `serve_telephony.py` first (it answers the
calls), then:

```bash
export ARI_BASE_URL=http://asterisk:8088 ARI_USER=ari ARI_PASSWORD=secret
python outbound_campaign.py contacts.json \
    --trunk telnyx --caller-id +15559876543 --max-concurrency 20 --delay 0.5
```

Preview first with `--dry-run` — the consent gate still runs, so you see who
would be dialed vs skipped, but no call is placed (no ARI creds needed):

```bash
python outbound_campaign.py contacts.json \
    --trunk telnyx --caller-id +15559876543 --dry-run
```

---

## 4. Scaling to 100+ concurrent calls

The media plane is cheap; **the GPU-bound STT/LLM/TTS is the real bottleneck.**
Separate them:

```
                 [Kamailio / OpenSIPS]        <- SIP proxy: registration,
                  /        |        \            routing, failover, load balance
        [Asterisk-1]  [Asterisk-2]  [Asterisk-N]  <- media servers (+ rtpengine
             \            |            /              for RTP relay/NAT)
              \___________|___________/
                          | AudioSocket / WS
                 [VoiceOS worker pool]         <- autoscaled; GPU for STT/TTS
                          |
                 [shared LLM endpoint(s)]      <- Ollama/vLLM/Groq, batched
```

- **Media sizing:** a single modern core handles *hundreds* of G.711↔L16
  transcoded calls; media is rarely the limit.
- **AI sizing is the constraint:** budget GPU per concurrent call for
  Whisper + TTS. Scale the VoiceOS worker pool independently of media servers;
  point them at a shared, batched LLM endpoint.
- **SIP proxy** (Kamailio/OpenSIPS) does trunk registration, distributes calls
  across media servers, and handles failover; **rtpengine** relays RTP.
- **Graceful drain:** stop accepting new AudioSocket connections, let active
  calls finish (`AudioSocketServer.stop()` closes the listener, not live calls).

---

## 5. Audio / codec / DTMF pitfalls checklist

- ☑️ **Sample rates:** telephony is **8 kHz**; VoiceOS is **16 kHz in / 24 kHz
  out**. Transcode both ways (`voiceos/telephony/transcode.py`), **stateful**
  per call/direction (avoids click artifacts at 20 ms frame boundaries).
- ☑️ **Codec:** AudioSocket carries **linear** 8k (`encoding="pcm16"`);
  Twilio/mod_audio_stream carry **mu-law** (`encoding="mulaw"`). Pick the right
  one — mismatch = static.
- ☑️ **Re-framing:** telephony frames are 20 ms (320 samples @16k) but Silero
  VAD needs **exactly 512-sample** frames — `MediaStreamTransport` re-chunks.
- ☑️ **DTMF is out-of-band:** the audio bridge is audio-only, so touch-tones
  arrive as media-server events (Asterisk `dtmf`/AudioSocket 0x03, RFC 4733),
  **not** in the audio. Capture them at the media-server layer.
- ☑️ **Barge-in:** `sink.interrupt()` stops outbound audio; also flush the
  media server's buffer (Twilio `clear` / stop streaming) for a clean cut.
- 🚨 **Python 3.13+:** stdlib `audioop` was removed (PEP 594) — `pip install
  audioop-lts`. Fine on 3.12.
- 🚨 **Outbound compliance (US):** AI voices are "artificial" under the TCPA
  (FCC Feb 2024) — outbound needs prior express consent. Set caller ID to a DID
  you own with STIR/SHAKEN attestation; add spend caps to prevent toll fraud.
```

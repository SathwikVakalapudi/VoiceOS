#!/usr/bin/env bash
# Render Asterisk config from environment at container start, then run Asterisk.
# Secrets (trunk password, ARI password) come from the environment (compose ->
# .env), so nothing sensitive is baked into the image or committed.
set -euo pipefail

CONF=/etc/asterisk
: "${TRUNK_NAME:=trunk}"
: "${SOFTPHONE_PASSWORD:=verysecret}"
: "${ARI_USERNAME:=ari}"
: "${ARI_PASSWORD:=change-me}"
: "${TRUNK_REGISTER:=true}"

# ---------------- pjsip.conf ----------------
{
  cat <<EOF
[transport-udp]
type=transport
protocol=udp
bind=0.0.0.0:5060
EOF
  # Behind NAT (home/laptop), advertise the public IP for media so the provider
  # can send RTP back. Set PUBLIC_IP in .env; also forward 10000-10100/udp.
  if [ -n "${PUBLIC_IP:-}" ]; then
    cat <<EOF
external_media_address=${PUBLIC_IP}
external_signaling_address=${PUBLIC_IP}
local_net=172.16.0.0/12
local_net=192.168.0.0/16
local_net=10.0.0.0/8
EOF
  fi

  # ---- Local softphone (for the 100/600 test extensions) ----
  cat <<EOF

[1001]
type=endpoint
context=voiceos-test
disallow=all
allow=ulaw,alaw
auth=1001-auth
aors=1001
direct_media=no
dtmf_mode=rfc4733

[1001-auth]
type=auth
auth_type=userpass
username=1001
password=${SOFTPHONE_PASSWORD}

[1001]
type=aor
max_contacts=1
remove_existing=yes
EOF

  # ---- Provider SIP trunk (only if TRUNK_SERVER is set) ----
  if [ -n "${TRUNK_SERVER:-}" ]; then
    cat <<EOF

[${TRUNK_NAME}-auth]
type=auth
auth_type=userpass
username=${TRUNK_USERNAME}
password=${TRUNK_PASSWORD}

[${TRUNK_NAME}]
type=aor
contact=sip:${TRUNK_SERVER}
qualify_frequency=30

[${TRUNK_NAME}]
type=endpoint
transport=transport-udp
context=voiceos-inbound
disallow=all
allow=ulaw,alaw
outbound_auth=${TRUNK_NAME}-auth
aors=${TRUNK_NAME}
from_user=${TRUNK_DID:-${TRUNK_USERNAME}}
from_domain=${TRUNK_SERVER}
direct_media=no
dtmf_mode=rfc4733
rtp_symmetric=yes
force_rport=yes
rewrite_contact=yes
EOF
    # Inbound matching: only needed to route INBOUND calls to this endpoint,
    # and it resolves at load time — a DNS blip would break config. Opt-in via
    # TRUNK_IDENTIFY_MATCH (the provider's signaling IP/CIDR or host). Outbound
    # origination does not need it.
    if [ -n "${TRUNK_IDENTIFY_MATCH:-}" ]; then
      cat <<EOF

[${TRUNK_NAME}]
type=identify
endpoint=${TRUNK_NAME}
match=${TRUNK_IDENTIFY_MATCH}
EOF
    fi
    if [ "${TRUNK_REGISTER}" = "true" ]; then
      cat <<EOF

[${TRUNK_NAME}-reg]
type=registration
transport=transport-udp
outbound_auth=${TRUNK_NAME}-auth
server_uri=sip:${TRUNK_SERVER}
client_uri=sip:${TRUNK_USERNAME}@${TRUNK_SERVER}
retry_interval=60
forbidden_retry_interval=300
expiration=3600
EOF
    fi
  fi
} > "${CONF}/pjsip.conf"

# ---------------- ari.conf ----------------
cat > "${CONF}/ari.conf" <<EOF
[general]
enabled=yes
pretty=yes

[${ARI_USERNAME}]
type=user
password=${ARI_PASSWORD}
EOF

echo "asterisk-entrypoint: trunk=${TRUNK_SERVER:-<none>} register=${TRUNK_REGISTER} did=${TRUNK_DID:-<unset>} ari_user=${ARI_USERNAME}"

# ARI's /ari HTTP routes don't mount reliably on first load (they register
# before the HTTP server is fully up). Reload res_ari once Asterisk is ready.
(
  for _ in $(seq 1 30); do
    if asterisk -rx "core show uptime" >/dev/null 2>&1; then
      asterisk -rx "module reload res_ari" >/dev/null 2>&1
      echo "asterisk-entrypoint: res_ari reloaded (/ari mounted)"
      break
    fi
    sleep 1
  done
) &

exec asterisk -f -vvv -T -W

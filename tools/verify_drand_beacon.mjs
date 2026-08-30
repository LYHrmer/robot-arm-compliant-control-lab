#!/usr/bin/env node

import {
  HttpCachingChain,
  HttpChainClient,
  fetchBeacon,
} from "drand-client";

const CHAIN = Object.freeze({
  hash: "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971",
  publicKey:
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c8" +
    "c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb5" +
    "ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a",
});

const RELAYS = Object.freeze([
  "https://api.drand.sh",
  "https://drand.cloudflare.com",
]);

function parseRound(value) {
  if (!/^\d+$/.test(value ?? "")) {
    throw new Error("round must be a positive integer");
  }
  const round = Number(value);
  if (!Number.isSafeInteger(round) || round < 1) {
    throw new Error("round is outside the JavaScript safe-integer range");
  }
  return round;
}

function clientFor(baseUrl) {
  const options = {
    disableBeaconVerification: false,
    noCache: true,
    chainVerificationParams: {
      chainHash: CHAIN.hash,
      publicKey: CHAIN.publicKey,
    },
  };
  const chainUrl = `${baseUrl}/${CHAIN.hash}`;
  const chain = new HttpCachingChain(chainUrl, options);
  const client = new HttpChainClient(chain, options);
  return { chainUrl, chain, client };
}

async function fetchAndVerify(baseUrl, round) {
  const { chainUrl, chain, client } = clientFor(baseUrl);
  const chainInfo = await chain.info();
  const beacon = await fetchBeacon(client, round);
  return { base_url: baseUrl, chain_url: chainUrl, chain_info: chainInfo, beacon };
}

async function observeLatestRounds() {
  return Promise.all(
    RELAYS.map(async (relay) => {
      const { client } = clientFor(relay);
      const latest = await client.latest();
      const verified = await fetchBeacon(client, latest.round);
      return verified.round;
    }),
  );
}

async function latestReferenceRound() {
  let observations = [];
  for (let attempt = 0; attempt < 3; attempt += 1) {
    observations = await observeLatestRounds();
    const skew = Math.max(...observations) - Math.min(...observations);
    if (skew <= 1) {
      return {
        round: Math.max(...observations),
        observedLatestRounds: observations,
      };
    }
  }
  const skew = Math.max(...observations) - Math.min(...observations);
  throw new Error(`official relays differ by ${skew} rounds after three attempts`);
}

async function main() {
  const requested = process.argv[2];
  const latest = requested === "latest" ? await latestReferenceRound() : null;
  const round = latest?.round ?? parseRound(requested);
  const responses = [];
  for (const relay of RELAYS) {
    responses.push(await fetchAndVerify(relay, round));
  }
  process.stdout.write(
    `${JSON.stringify({
      verifier: "drand-client-js",
      verifier_version: "1.4.2",
      cryptographic_signature_verified: true,
      mode: latest ? "latest-reference" : "exact-round",
      round,
      ...(latest
        ? { observed_latest_rounds: latest.observedLatestRounds }
        : {}),
      responses,
    })}\n`,
  );
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error}\n`);
  process.exitCode = 1;
});

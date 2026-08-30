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

async function fetchAndVerify(baseUrl, round) {
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
  const chainInfo = await chain.info();
  const beacon = await fetchBeacon(client, round);
  return { base_url: baseUrl, chain_url: chainUrl, chain_info: chainInfo, beacon };
}

async function main() {
  const round = parseRound(process.argv[2]);
  const responses = [];
  for (const relay of RELAYS) {
    responses.push(await fetchAndVerify(relay, round));
  }
  process.stdout.write(
    `${JSON.stringify({
      verifier: "drand-client-js",
      verifier_version: "1.4.2",
      cryptographic_signature_verified: true,
      round,
      responses,
    })}\n`,
  );
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error}\n`);
  process.exitCode = 1;
});

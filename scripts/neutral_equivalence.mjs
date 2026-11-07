// The neutral package gives the samples of the package it was made from, bit for bit.
//
//   node scripts/neutral_equivalence.mjs <original-dir> <neutral-dir> <name-map.json>
//
// Both exported calls run through both packages over materials, speeds and sample rates. Every
// array must be equal byte by byte, and the sounding ones must hold samples that are not zero.
// Exit 0 when equal, 1 when not, 2 when nothing could be compared.

import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [origDir, neutralDir, mapFile] = process.argv.slice(2);
if (!origDir || !neutralDir || !mapFile) {
  console.error("usage: neutral_equivalence.mjs <original-dir> <neutral-dir> <name-map.json>");
  process.exit(2);
}
const map = JSON.parse(fs.readFileSync(mapFile, "utf8")).api;

async function load(dir, base) {
  const glue = await import(pathToFileURL(path.join(dir, `${base}.js`)).href);
  glue.initSync({ module: fs.readFileSync(path.join(dir, `${base}${base === "p" ? "_bg" : ""}.wasm`)) });
  return glue;
}
const glueFile = fs.readdirSync(neutralDir).find((f) => f.endsWith(".js"));
if (!glueFile) {
  console.error("no glue in the neutral package");
  process.exit(2);
}
const orig = await load(origDir, "p");
const neutral = await load(neutralDir, glueFile.slice(0, -3));

const CALLS = [];
for (const rate of [22050, 44100, 48000]) {
  for (const material of [0, 1]) {
    for (const speed of [0, 0.5, 3, 6, 40]) {
      CALLS.push(["renderImpactAtSpeed", [material, speed, rate], speed > 0]);
    }
  }
  CALLS.push(["renderImpact", [880, 0.5, 0.2, 0.5, 0, 0.3, 0.02, 3000, rate], true]);
  CALLS.push(["renderImpact", [110, 0.8, 0.1, 0.4, 1, 0.1, 0.01, 800, rate], true]);
}
CALLS.push(["renderImpactAtSpeed", [9, 3, 48000], false]);

let compared = 0;
let sounding = 0;
let samples = 0;
let bad = 0;
for (const [name, args, sounds] of CALLS) {
  const renamed = map[name];
  if (typeof orig[name] !== "function" || typeof neutral[renamed] !== "function") {
    console.error(`${name} is not exported by both packages`);
    process.exit(2);
  }
  const a = orig[name](...args);
  const b = neutral[renamed](...args);
  const ab = Buffer.from(a.buffer, a.byteOffset, a.byteLength);
  const bb = Buffer.from(b.buffer, b.byteOffset, b.byteLength);
  compared += 1;
  samples += a.length;
  if (!ab.equals(bb)) {
    console.error(`${name}(${args.join(", ")}) differs`);
    bad += 1;
  }
  if (sounds) {
    if (a.some((x) => x !== 0)) sounding += 1;
    else {
      console.error(`${name}(${args.join(", ")}) is silent`);
      bad += 1;
    }
  }
}
if (compared === 0 || samples === 0 || sounding === 0) {
  console.error("nothing was compared");
  process.exit(2);
}
console.log(`neutral equivalence: ${compared} calls, ${sounding} sounding, ${samples} samples, ${bad} bad`);
process.exit(bad === 0 ? 0 : 1);

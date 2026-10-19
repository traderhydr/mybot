// Reads a prompt on stdin, asks Claude via Puter.js, prints the reply text.
// Needs PUTER_AUTH_TOKEN (get one with: node puter_ai.cjs --login).
const { init, getAuthToken } = require("@heyputer/puter.js/src/init.cjs");

(async () => {
  if (process.argv.includes("--login")) {
    console.log(await getAuthToken()); // opens a browser; run on a machine that has one
    return;
  }
  const token = process.env.PUTER_AUTH_TOKEN;
  if (!token) throw new Error("PUTER_AUTH_TOKEN not set");
  let prompt = "";
  for await (const chunk of process.stdin) prompt += chunk;
  const puter = init(token);
  const res = await puter.ai.chat(prompt, { model: process.env.PUTER_MODEL || "claude-sonnet-5-5" });
  const c = res?.message?.content;
  process.stdout.write(typeof c === "string" ? c : (c || []).map((p) => p.text || "").join(""));
})().catch((e) => { console.error(e?.message || e); process.exit(1); });

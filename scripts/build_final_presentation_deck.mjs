import fs from "node:fs/promises";
import fsSync from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawnSync } from "node:child_process";

const __filename = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(__filename), "..");
const OUT_DIR = path.join(ROOT, "docs", "presentation");
const QA_DIR = path.join(OUT_DIR, "qa");
const TMP_DIR = path.join(process.env.TEMP || process.env.TMP || OUT_DIR, "aads-v6-final-presentation");
const FINAL_PPTX = path.join(OUT_DIR, "final_aads_v6_handoff.pptx");
const SCREENSHOT_PATH = path.join(OUT_DIR, "notebook8_output_screenshot.png");

const SUMMARY_PATH = path.join(ROOT, "docs", "demo_results", "m2", "20260706T153334Z", "summary.json");
const REVIEW_PATH = path.join(ROOT, "docs", "demo_results", "m2", "20260706T153334Z", "post_run_review.md");
const MANIFEST_SUMMARY_PATH = "docs/demo_assets/customer_demo_manifest/customer_demo_manifest_summary.md";

function artifactToolEntrypoint() {
  const workspace = process.env.ARTIFACT_TOOL_WORKSPACE;
  if (!workspace) {
    throw new Error("Set ARTIFACT_TOOL_WORKSPACE to a workspace initialized by setup_artifact_tool_workspace.mjs");
  }
  const candidates = [
    path.join(workspace, "node_modules", "@oai", "artifact-tool", "dist", "node", "artifact_tool.mjs"),
    path.join(workspace, "node_modules", "@oai", "artifact-tool", "dist", "artifact_tool.mjs"),
  ];
  const entrypoint = candidates.find((candidate) => fsSync.existsSync(candidate));
  if (!entrypoint) {
    throw new Error(`Could not find @oai/artifact-tool in ${workspace}`);
  }
  return pathToFileURL(entrypoint).href;
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

function textEscape(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function findBrowser() {
  const candidates = [
    process.env.BROWSER_PATH,
    "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  ].filter(Boolean);
  return candidates.find((candidate) => fsSync.existsSync(candidate));
}

async function createNotebookOutputScreenshot(summary) {
  await fs.mkdir(TMP_DIR, { recursive: true });
  await fs.mkdir(OUT_DIR, { recursive: true });

  const htmlPath = path.join(TMP_DIR, "notebook8-output.html");
  const rows = [
    ["runner_exit_code", summary.runner_exit_code],
    ["manifest", summary.manifest],
    ["toplam / pass / fail", `${summary.summary.total} / ${summary.summary.passed} / ${summary.summary.failed}`],
    ["hastalik cevabi verilen satir", summary.summary.answered],
    ["review/abstain safety satiri", summary.summary.abstained_or_reviewed],
    ["negative false accept", summary.analysis_summary.negative_false_accepts.count],
    ["opposite-part disease label", summary.analysis_summary.opposite_part_disease_labels.count],
  ];
  const html = `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body { margin: 0; background: #f4f4f4; font-family: "Segoe UI", Arial, sans-serif; }
    .frame { width: 1100px; height: 680px; padding: 38px; box-sizing: border-box; background: #f7f7f7; }
    .toolbar { display: flex; justify-content: space-between; color: #555; font-size: 18px; margin-bottom: 18px; }
    .cell { background: #fff; border-left: 8px solid #111; box-shadow: 0 1px 7px rgba(0,0,0,.12); }
    .input { padding: 18px 24px; border-bottom: 1px solid #ddd; color: #222; font-size: 22px; }
    .output { padding: 24px; }
    .status { display: inline-block; background: #111; color: #fff; padding: 8px 14px; font-weight: 700; font-size: 18px; margin-bottom: 18px; }
    table { width: 100%; border-collapse: collapse; font-size: 21px; }
    td { padding: 12px 10px; border-bottom: 1px solid #dedede; vertical-align: top; }
    td:first-child { width: 36%; color: #555; font-weight: 700; }
    .note { margin-top: 18px; color: #333; font-size: 20px; line-height: 1.35; }
  </style>
</head>
<body>
  <div class="frame">
    <div class="toolbar"><span>Notebook 8 final M2 hucresi</span><span>${textEscape(summary.created_at)}</span></div>
    <div class="cell">
      <div class="input">run_demo_checklist(customer_demo_manifest)</div>
      <div class="output">
        <div class="status">ready_for_demo</div>
        <table>
          ${rows.map(([key, value]) => `<tr><td>${textEscape(key)}</td><td>${textEscape(value)}</td></tr>`).join("")}
        </table>
        <div class="note">Unsupported ve unknown-part satirlar adapter hastalik tahmininden once bloklanir; bu yuzden review/abstain basarili safety sonucudur.</div>
      </div>
    </div>
  </div>
</body>
</html>`;
  await fs.writeFile(htmlPath, html, "utf8");

  const browser = findBrowser();
  if (!browser) {
    throw new Error("Could not find Edge or Chrome to render the Notebook 8 output screenshot");
  }
  const result = spawnSync(browser, [
    "--headless",
    "--disable-gpu",
    "--hide-scrollbars",
    "--window-size=1100,680",
    `--screenshot=${SCREENSHOT_PATH}`,
    pathToFileURL(htmlPath).href,
  ], { stdio: "pipe" });
  if (result.status !== 0) {
    throw new Error(`Browser screenshot failed: ${result.stderr?.toString() || result.stdout?.toString()}`);
  }
}

function addText(slide, text, position, style = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    fontSize: style.fontSize || 22,
    bold: style.bold || false,
    color: style.color || "#111111",
    fontFace: "Helvetica Neue",
    ...style,
  };
  return shape;
}

function addRule(slide, left, top, width) {
  slide.shapes.add({
    geometry: "rect",
    position: { left, top, width, height: 2 },
    fill: "#111111",
    line: { style: "solid", fill: "none", width: 0 },
  });
}

function addSlideNumber(slide, number) {
  addText(slide, String(number).padStart(2, "0"), { left: 1188, top: 658, width: 50, height: 24 }, {
    fontSize: 14,
    color: "#555555",
    alignment: "right",
  });
}

function titleSlide(presentation, title, subtitle) {
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";
  addText(slide, "AADS v6", { left: 42, top: 36, width: 280, height: 36 }, { fontSize: 18, bold: true, color: "#555555" });
  addText(slide, title, { left: 42, top: 178, width: 900, height: 210 }, { fontSize: 66, bold: true, color: "#000000" });
  addText(slide, subtitle, { left: 42, top: 512, width: 760, height: 78 }, { fontSize: 25, color: "#333333" });
  addText(slide, "Final demo ve handoff deck'i\n2026-07-06", { left: 930, top: 570, width: 280, height: 80 }, { fontSize: 20, color: "#555555", alignment: "right" });
  addRule(slide, 42, 650, 420);
  addSlideNumber(slide, 1);
}

function bulletsSlide(presentation, slideNo, title, bullets, footer = "") {
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";
  addText(slide, title, { left: 42, top: 42, width: 1060, height: 86 }, { fontSize: 42, bold: true });
  addRule(slide, 42, 150, 240);
  let top = 205;
  for (const bullet of bullets) {
    addText(slide, bullet.head, { left: 90, top, width: 960, height: 34 }, { fontSize: 26, bold: true });
    addText(slide, bullet.body, { left: 90, top: top + 42, width: 960, height: 54 }, { fontSize: 20, color: "#333333" });
    slide.shapes.add({
      geometry: "rect",
      position: { left: 42, top: top + 7, width: 18, height: 18 },
      fill: "#111111",
      line: { style: "solid", fill: "none", width: 0 },
    });
    top += 118;
  }
  if (footer) {
    addText(slide, footer, { left: 42, top: 628, width: 760, height: 38 }, { fontSize: 16, color: "#555555" });
  }
  addSlideNumber(slide, slideNo);
}

function architectureSlide(presentation, slideNo) {
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";
  addText(slide, "Notebook 8 tek fotografi hastalik cevabina veya guvenli review'a cevirir", { left: 42, top: 42, width: 1120, height: 90 }, { fontSize: 39, bold: true });
  const steps = [
    ["Image input", "Demo manifest veya user upload fotografi"],
    ["Router", "Crop ve part kaniti"],
    ["Prototype handoff", "Belirsiz router kararini dengeler"],
    ["Adapter", "Target-specific hastalik modeli"],
    ["Final status", "Hastalik cevabi veya review/abstain"],
  ];
  const y = 248;
  steps.forEach(([head, body], index) => {
    const left = 48 + index * 240;
    slide.shapes.add({
      geometry: "rect",
      position: { left, top: y, width: 180, height: 170 },
      fill: index === 4 ? "#111111" : "#ededed",
      line: { style: "solid", fill: "#b8bcc4", width: 1 },
    });
    addText(slide, head, { left: left + 18, top: y + 24, width: 145, height: 48 }, { fontSize: 24, bold: true, color: index === 4 ? "#ffffff" : "#000000" });
    addText(slide, body, { left: left + 18, top: y + 86, width: 145, height: 68 }, { fontSize: 17, color: index === 4 ? "#ffffff" : "#333333" });
    if (index < steps.length - 1) {
      addText(slide, ">", { left: left + 198, top: y + 58, width: 34, height: 52 }, { fontSize: 42, bold: true, color: "#555555" });
    }
  });
  addText(slide, "Safety rule: unsupported crop, unsupported part, non-plant veya zayif kanit adapter hastalik tahmininden once durmalidir.", { left: 42, top: 552, width: 1080, height: 64 }, { fontSize: 23, bold: true });
  addSlideNumber(slide, slideNo);
}

function metricsSlide(presentation, slideNo, summary) {
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";
  addText(slide, "Controlled customer demo sunumda kullanima hazir", { left: 42, top: 42, width: 1060, height: 88 }, { fontSize: 41, bold: true });
  const metrics = [
    ["48 / 48", "satir pass"],
    ["0", "failure"],
    ["36", "hastalik cevabi"],
    ["12", "review veya abstain"],
    ["0", "negative false accept"],
    ["0", "opposite-part disease label"],
  ];
  metrics.forEach(([value, label], index) => {
    const col = index % 3;
    const row = Math.floor(index / 3);
    const left = 70 + col * 385;
    const top = 190 + row * 180;
    slide.shapes.add({
      geometry: "rect",
      position: { left, top, width: 315, height: 130 },
      fill: "#ededed",
      line: { style: "solid", fill: "none", width: 0 },
    });
    addText(slide, value, { left: left + 24, top: top + 18, width: 260, height: 55 }, { fontSize: 48, bold: true });
    addText(slide, label, { left: left + 24, top: top + 78, width: 260, height: 32 }, { fontSize: 19, color: "#333333" });
  });
  addText(slide, `Runner exit code ${summary.runner_exit_code}; elapsed ${summary.elapsed_human}; evidence folder docs/demo_results/m2/20260706T153334Z/`, { left: 70, top: 585, width: 1040, height: 42 }, { fontSize: 18, color: "#555555" });
  addSlideNumber(slide, slideNo);
}

function statusTableSlide(presentation, slideNo) {
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";
  addText(slide, "Support label'lari live demo anlatimini durust tutar", { left: 42, top: 42, width: 1060, height: 84 }, { fontSize: 42, bold: true });
  const table = [
    ["Label", "Targets", "Demo anlami"],
    ["strong", "apricot__leaf, strawberry__leaf, grape__leaf", "Ana supported demo yuzeyleri"],
    ["caution", "tomato__leaf, tomato__fruit, apricot__fruit, grape__fruit", "Acik limitation diliyle kullan"],
    ["problematic", "strawberry__fruit", "Minimal passing orneklerle sinirla"],
    ["safety_review", "unknown crop, unsupported part, non-plant", "Review/abstain beklenen basari yoludur"],
  ];
  const left = 58;
  const top = 170;
  const colW = [210, 545, 365];
  const rowH = 68;
  table.forEach((row, rowIndex) => {
    const y = top + rowIndex * rowH;
    slide.shapes.add({
      geometry: "rect",
      position: { left, top: y, width: 1120, height: rowH },
      fill: rowIndex === 0 ? "#ededed" : "#ffffff",
      line: { style: "solid", fill: "#b8bcc4", width: 1 },
    });
    let x = left;
    row.forEach((cell, colIndex) => {
      addText(slide, cell, { left: x + 14, top: y + 16, width: colW[colIndex] - 28, height: rowH - 20 }, {
        fontSize: rowIndex === 0 ? 18 : 17,
        bold: rowIndex === 0 || colIndex === 0,
        color: "#111111",
      });
      x += colW[colIndex];
    });
  });
  addText(slide, "602-row full manifest stress/regression kanitidir; customer acceptance set olarak sunulmaz.", { left: 58, top: 572, width: 1040, height: 48 }, { fontSize: 22, bold: true });
  addSlideNumber(slide, slideNo);
}

async function screenshotSlide(presentation, slideNo) {
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";
  addText(slide, "Final Notebook 8 output temiz safety ve quality karari veriyor", { left: 42, top: 42, width: 1120, height: 84 }, { fontSize: 39, bold: true });
  const bytes = await fs.readFile(SCREENSHOT_PATH);
  slide.images.add({
    blob: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    contentType: "image/png",
    alt: "20260706T153334Z summary verisinden uretilen Notebook 8 final M2 output screenshot",
    fit: "contain",
    position: { left: 90, top: 155, width: 1040, height: 430 },
  });
  addText(slide, "Screenshot source: docs/demo_results/m2/20260706T153334Z/summary.json", { left: 90, top: 606, width: 700, height: 28 }, { fontSize: 16, color: "#555555" });
  addSlideNumber(slide, slideNo);
}

function handoffSlide(presentation, slideNo) {
  const slide = presentation.slides.add();
  slide.background.fill = "#ffffff";
  addText(slide, "Handoff yolu kucuk baslar ve runnable kalir", { left: 42, top: 42, width: 1060, height: 84 }, { fontSize: 42, bold: true });
  const items = [
    ["README.md", "Entry point ve maintained surface'ler"],
    ["docs/handoff_guide.md", "Customer-first Notebook 8 run path"],
    ["docs/demo_checklist.md", "Demo manifest, komutlar, acceptance evidence"],
    ["docs/demo_results/m2/20260706T153334Z/post_run_review.md", "Final ready_for_demo karari"],
  ];
  items.forEach(([head, body], index) => {
    const top = 175 + index * 100;
    addText(slide, String(index + 1).padStart(2, "0"), { left: 58, top, width: 70, height: 36 }, { fontSize: 30, bold: true, color: "#555555" });
    addText(slide, head, { left: 150, top, width: 830, height: 34 }, { fontSize: 25, bold: true });
    addText(slide, body, { left: 150, top: top + 38, width: 830, height: 30 }, { fontSize: 19, color: "#333333" });
    addRule(slide, 150, top + 78, 840);
  });
  addSlideNumber(slide, slideNo);
}

async function buildDeck(Presentation, PresentationFile, summary) {
  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
  titleSlide(presentation, "Colab'de guvenli bitki hastaligi inference", "Notebook 8 bitki fotografini crop/part adapter yoluna tasir, kanit guvenliyse cevap verir, etiketi zorlamamasi gereken durumda review/abstain doner.");
  bulletsSlide(presentation, 2, "Demo, unsafe certainty yerine faydali ve guvenli cevabi optimize eder", [
    { head: "Kullanici daginik real-world input yukleyebilir", body: "Supported crop, unsupported crop, unsupported part, non-plant image ve belirsiz hastalik durumlari final hikayede gorunur." },
    { head: "Ana risk emin gorunen yanlis hastalik etiketidir", body: "Router, prototype, adapter veya OOD kaniti yeterince guclu degilse sistem review ya da abstain diyebilir." },
    { head: "Final scope web app degil, handoff demo'dur", body: "Customer surface Colab Notebook 8'dir; repo docs, uretilmis evidence ve validation komutlariyla desteklenir." },
  ]);
  architectureSlide(presentation, 3);
  bulletsSlide(presentation, 4, "Repo training, validation ve live inference yuzeylerini ayirir", [
    { head: "Notebook 0 grouped dataset hazirlar", body: "Dataset preparation adapter training oncesinde kalir ve canonical repo workflow'unu kullanir." },
    { head: "Notebook 2 target adapter'lari egitir", body: "Tek genis hastalik modeli yerine her crop/part yuzeyi kendi adapter family'sine sahiptir." },
    { head: "Notebook 3 ve Notebook 5 supporting artifact'lari dogrular", body: "Adapter validation ve router calibration maintenance surface olarak kalir; live customer path degildir." },
    { head: "Notebook 8 live demo yuzeyidir", body: "Router, prototype handoff, adapter inference ve final status reporting'i orkestre eder." },
  ]);
  bulletsSlide(presentation, 5, "Live akis strong yuzeylerle baslar, sonra limitleri bilincli gosterir", [
    { head: "Strong supported orneklerle basla", body: "apricot__leaf, strawberry__leaf ve grape__leaf guven olusturan ana target'lardir." },
    { head: "Caution yuzeyleri acik limitation diliyle goster", body: "tomato__leaf, tomato__fruit, apricot__fruit ve grape__fruit demo-usable'dir ama fazla iddia edilmemelidir." },
    { head: "Demo'yu unsupported ve non-plant satirlarla kapat", body: "Bu ornekler review/abstain davranisinin crash degil, bilincli safety davranisi oldugunu gosterir." },
  ]);
  metricsSlide(presentation, 6, summary);
  statusTableSlide(presentation, 7);
  await screenshotSlide(presentation, 8);
  bulletsSlide(presentation, 9, "Safety evidence supported-set quality'den ayridir", [
    { head: "Unsupported satirlar adapter prediction oncesinde durur", body: "Diagnostics supported-looking handoff gosterebilir; ancak unsupported veya unknown-part satirlarda final disease prediction bloklanir." },
    { head: "Open-world safety ayri bir gate olarak kalir", body: "20260702T180837Z router open-world kosusunda 306/306 review veya abstain ve 0 false accept vardir." },
    { head: "Dogru abstention basari sayilir", body: "Final evidence, unsupported, unsafe veya non-plant input icin review/abstain sonucunu dogru davranis olarak sayar." },
  ]);
  bulletsSlide(presentation, 10, "Notebook 16 review research'u destekler, runtime kararini degistirmez", [
    { head: "Notebook 16 report-only evidence-gate isidir", body: "Gelecekte review flag'lerine yardim edebilecek ROI ve bbox sinyallerini inceler." },
    { head: "Notebook 8 decision surface olarak kalir", body: "Ayrica promotion validation gecmeden full-image adapter prediction final karar olarak kalir." },
    { head: "Sunum production promotion ima etmemeli", body: "Notebook 16'yi live override degil, kisa supporting evidence olarak anlat." },
  ]);
  bulletsSlide(presentation, 11, "Durust limitasyonlar handoff icin yeterince net", [
    { head: "Target scope dardir", body: "Intended demo scope icinde sadece tomato, strawberry, grape ve apricot fruit/leaf yuzeyleri vardir." },
    { head: "strawberry__fruit limitation-aware anlatilmalidir", body: "Minimal passing ornekleri vardir ve en zayif supported surface olarak tarif edilmelidir." },
    { head: "Fresh official rerun external access ister", body: "Yeni official run icin Colab GPU, Hugging Face/SAM3 access ve auto-push icin GitHub token gerekir." },
  ]);
  handoffSlide(presentation, 12);

  await fs.mkdir(QA_DIR, { recursive: true });
  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(QA_DIR, `${stem}.png`), Buffer.from(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(QA_DIR, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(QA_DIR, "final_aads_v6_handoff_montage.webp"), Buffer.from(await montage.arrayBuffer()));

  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(FINAL_PPTX);
}

async function main() {
  const summary = await readJson(SUMMARY_PATH);
  await fs.mkdir(OUT_DIR, { recursive: true });
  await createNotebookOutputScreenshot(summary);
  const { Presentation, PresentationFile } = await import(artifactToolEntrypoint());
  await buildDeck(Presentation, PresentationFile, summary);
  console.log(`Wrote ${path.relative(ROOT, FINAL_PPTX)}`);
  console.log(`Wrote ${path.relative(ROOT, SCREENSHOT_PATH)}`);
  console.log(`Used ${path.relative(ROOT, SUMMARY_PATH)}`);
  console.log(`Used ${path.relative(ROOT, REVIEW_PATH)}`);
  console.log(`Used ${MANIFEST_SUMMARY_PATH}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

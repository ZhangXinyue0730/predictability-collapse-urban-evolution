import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const IN = process.env.SUBMISSION_AUDIT_XLSX || 'outputs/Nature投稿包核对表_中文版.xlsx';
const PREVIEW_DIR = process.env.SUBMISSION_AUDIT_PREVIEW_DIR || 'outputs/submission_audit_preview';

const blob = await FileBlob.load(IN);
const wb = await SpreadsheetFile.importXlsx(blob);

const sheets = await wb.inspect({ kind: 'sheet', include: 'id,name', maxChars: 5000 });
console.log('SHEETS');
console.log(sheets.ndjson);

const errors = await wb.inspect({
  kind: 'match',
  searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A',
  options: { useRegex: true, maxResults: 300 },
  summary: 'formula error scan',
  maxChars: 4000,
});
console.log('ERROR_SCAN');
console.log(errors.ndjson || '(none)');

await fs.mkdir(PREVIEW_DIR, { recursive: true });
for (const sheet of ['01_投稿包总览', '02_主文核查', '03_图表与源数据对应']) {
  const png = await wb.render({ sheetName: sheet, autoCrop: 'all', scale: 1, format: 'png' });
  const bytes = new Uint8Array(await png.arrayBuffer());
  await fs.writeFile(`${PREVIEW_DIR}/${sheet}.png`, bytes);
}
console.log(PREVIEW_DIR);

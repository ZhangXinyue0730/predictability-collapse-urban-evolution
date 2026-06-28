import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const analysisDir =
  process.env.ANALYSIS_DIR || "outputs/national_analysis_dynamic_t1";
const tableCsv = `${analysisDir}/appendix_A_table_A1_predictability_by_classifier.csv`;
const detailCsv = `${analysisDir}/appendix_A_multimodel_auc_city_period.csv`;
const outXlsx = `${analysisDir}/appendix_A_table_A1_predictability_by_classifier.xlsx`;

const tableText = await fs.readFile(tableCsv, "utf8");
const detailText = await fs.readFile(detailCsv, "utf8");

const workbook = await Workbook.fromCSV(tableText, { sheetName: "Table A1" });
await workbook.fromCSV(detailText, { sheetName: "city_period_auc" });

for (const sheetName of ["Table A1", "city_period_auc"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange();
  used.format.borders = { preset: "all", style: "thin", color: "#D9D9D9" };
  const header = sheet.getRangeByIndexes(0, 0, 1, used.columnCount);
  header.format = {
    fill: "#1F4E79",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
  };
  used.format.autofitColumns();
  used.format.autofitRows();
}

const tableSheet = workbook.worksheets.getItem("Table A1");
tableSheet.getRange("B:D").format.numberFormat = "0.000";
tableSheet.getRange("E:E").format.numberFormat = "0.0";
tableSheet.getRange("F:G").format.numberFormat = "0.000";

const detailSheet = workbook.worksheets.getItem("city_period_auc");
detailSheet.getRange("K:K").format.numberFormat = "0.0000";
detailSheet.getRange("J:J").format.numberFormat = "0.00%";

const inspect = await workbook.inspect({
  kind: "table",
  range: "Table A1!A1:I4",
  include: "values",
  tableMaxRows: 5,
  tableMaxCols: 10,
});
console.log(inspect.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outXlsx);
console.log(outXlsx);

import fs from 'node:fs/promises';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const OUT_DIR = process.env.SUBMISSION_AUDIT_DIR || 'outputs/submission_audit';
const OUT = `${OUT_DIR}/Nature投稿包核对表_中文版.xlsx`;

const wb = Workbook.create();

function addSheet(name, headers, rows, widths = []) {
  const ws = wb.worksheets.add(name);
  ws.showGridLines = false;

  const titleRange = ws.getRangeByIndexes(0, 0, 1, headers.length);
  titleRange.merge();
  titleRange.values = [[name]];
  titleRange.format.fill.color = '#1F4E78';
  titleRange.format.font.color = '#FFFFFF';
  titleRange.format.font.bold = true;
  titleRange.format.font.size = 14;
  titleRange.format.horizontalAlignment = 'center';
  titleRange.format.verticalAlignment = 'center';
  titleRange.format.rowHeight = 30;

  const hdr = ws.getRangeByIndexes(2, 0, 1, headers.length);
  hdr.values = [headers];
  hdr.format.fill.color = '#D9EAF7';
  hdr.format.font.bold = true;
  hdr.format.font.color = '#17365D';
  hdr.format.borders = { preset: 'all', style: 'thin', color: '#A6A6A6' };
  hdr.format.wrapText = true;
  hdr.format.verticalAlignment = 'center';
  hdr.format.horizontalAlignment = 'center';

  if (rows.length) {
    const body = ws.getRangeByIndexes(3, 0, rows.length, headers.length);
    body.values = rows;
    body.format.borders = { preset: 'all', style: 'thin', color: '#D9D9D9' };
    body.format.wrapText = true;
    body.format.verticalAlignment = 'top';
  }

  for (let i = 0; i < headers.length; i++) {
    const col = ws.getRangeByIndexes(0, i, rows.length + 5, 1);
    col.format.columnWidth = widths[i] || 18;
  }
  ws.freezePanes.freezeRows(3);
  return ws;
}

const overview = [
  ['主文稿', 'Predictability_Collapse_in_Urban_Evolution_EN.docx', '当前英文主文基础稿', '基本可用，需小修', '以后以这份英文稿作为主文基础，不再以中文稿作为直接投稿版本。'],
  ['补充信息', 'Supplementary_Information.docx', 'Supplementary Information 的初稿/模板', '可用，但需清理', '删除模板化表达，统一补充表编号，并把长方法细节放入 SI。'],
  ['补充数据', 'Supplementary_Data.xlsx', '大型数值矩阵和特征字典', '待整理/待核验', '建议放入79城×8期矩阵、置换重要性、转移矩阵、句法汇总、167维特征字典等。'],
  ['Source Data', 'Source_Data.xlsx', '支撑每张主图和扩展数据图的源数据', '待整理', 'Nature通常要求每个图/图板对应可追溯的数据表。'],
  ['Reporting Summary', 'Reporting Summary表格/文档', '投稿合规文件', '后续准备', '等图表、数据、方法和代码可用性声明锁定后再填写。'],
  ['Data Availability', '主文中的数据可用性声明', '已有初稿', '需具体化', '明确TCULU、OSM、统计年鉴、衍生数据的来源和共享方式。'],
  ['Code Availability', '主文中的代码可用性声明', '已有初稿', '需具体化', '说明pipeline、绘图脚本、核验脚本是否归档、开源或依请求提供。'],
];
addSheet('01_投稿包总览', ['材料类型','当前文件 / 目标文件','用途','状态','下一步动作'], overview, [24,44,38,20,72]);

const mainAudit = [
  ['标题', 'Predictability collapse in urban evolution', '约42个字符', '可以保留', '暂时不需要修改。'],
  ['摘要', 'Abstract标题后的主段落', '粗略统计约158词', '可能需压缩', '如果Nature Cities Article摘要限制为150词，需要压缩约10-15词。'],
  ['关键词', '包含 White-box physical models', '白盒已经不作为当前主文重点', '建议修改', '改为 Built-to-built audit / Redevelopment diagnostics，或直接删除该关键词。'],
  ['主图', 'Fig. 1-Fig. 4', '4张主图', '基本合理', '符合Nature主文图数量较精简的要求。'],
  ['扩展数据图', 'Extended Data Fig. 1-Fig. 5', '5张支撑图', '需逐图核对', '每张Extended Data图都要有对应Source Data表。'],
  ['主文表格', '目前正文末尾有3张表', '更适合作为Supplementary Table', '需确认放置位置', '如果放入SI，应从主文正文表格区移除或统一编号。'],
  ['Methods', 'Methods章节', '粗略约768词', '长度可控', '更长的算法细节放到Supplementary Notes中。'],
  ['参考文献', '38篇', '数量合理', '需核对', '重点核对TCULU、Han 2020、Nature Cities城市层级三篇是否准确。'],
  ['白盒残留表述', 'Built-to-built (white-box) audit 等位置', '与当前主文去白盒方向不完全一致', '必须修正', '建议统一改为 Built-to-built audit / Greenfield audit / diagnostic audit。'],
  ['补充表编号', '主文引用 Supplementary Table 3 为M0-M7', 'SI中M0-M7是 Supplementary Table 2', '必须修正', '统一主文、图注、SI三处编号。'],
];
addSheet('02_主文核查', ['章节/项目','位置','当前情况','状态','修改建议'], mainAudit, [24,40,50,20,72]);

const figRows = [
  ['Fig. 1', '城市更新可预测性的跨模型衰减', 'AUC衰减：GBDT/RF/L2-LR', '主文', 'Fig1a/Fig1b/Fig1c', '核对是否与多模型AUC稳健性表一致。'],
  ['Fig. 2', '扩张面积、更新面积与更新主导度', 'Expansion-renewal crossover', '主文', 'Fig2各图板', '必须使用最终 non-vacant dynamic_t1 口径。'],
  ['Fig. 3', '分时期置换重要性与驱动因子转换', '特征族和关键变量变化', '主文', 'Fig3各图板', '必须写 permutation importance，不写传统SHAP。'],
  ['Fig. 4', '内部更新与外围扩张的可预测性对比', 'Built-to-built vs greenfield vs one-step baseline', '主文', 'Fig4', '去掉white-box表述，改为audit/diagnostic。'],
  ['Extended Data Fig. 1', 'AP跨模型衰减', 'AP稳健性结果', '扩展数据', 'EDF1', '核对AP稳健性输出数据。'],
  ['Extended Data Fig. 2', 'M0-M7消融实验', 'M0-M7 AUC/AP', '扩展数据', 'EDF2', '特征组合定义应指向Supplementary Table。'],
  ['Extended Data Fig. 3', '2020-2024关键变量PDP曲线', '偏依赖分析', '扩展数据', 'EDF3', '注意图号不要与中文旧稿冲突。'],
  ['Extended Data Fig. 4', '更新主导度异质性与市政转入比例', '城市层面热图/市政转入', '扩展数据', 'EDF4', '核对城市排序和Source Data。'],
  ['Extended Data Fig. 5', '区域差异与Kruskal-Wallis检验', '区域统计检验', '扩展数据', 'EDF5', '完整H值、p值和组内中位数放入Supplementary Data。'],
];
addSheet('03_图表与源数据对应', ['图号','图题/内容','核心数据','放置位置','需要的Source Data表','核查/动作'], figRows, [22,46,38,20,34,64]);

const suppRows = [
  ['Supplementary Table 1', '主文', '多模型AUC稳健性', '主文末尾表格区', '可作为SI Table 1', '如果移入SI，需要同步主文引用。'],
  ['Supplementary Table 2', '主文', 'AUC与社会经济变量的TWFE回归', '主文末尾表格区', '可作为SI Table 2或Extended Data Table', '系数和p值必须与当前输出一致。'],
  ['Supplementary Table 3', '主文', 'M0-M7特征组合设置', '主文末尾表格区', '目前与SI编号冲突', '主文写Table 3，但SI里M0-M7是Table 2。'],
  ['Supplementary Table 1', 'SI文件', '用地类型编码表', 'SI正文中', '可保留为SI Table 1', '有助于解释1-6和0类。'],
  ['Supplementary Table 2', 'SI文件', 'M0-M7特征组合定义', 'SI正文中', '可保留为SI Table 2', '与主文M0-M7引用编号冲突。'],
  ['Supplementary Table 3', 'SI文件', '特征族字典汇总', 'SI正文中', '只是汇总表', '完整167维特征字典应放入Supplementary Data Excel。'],
  ['Supplementary Table 4', 'SI文件', '区域Kruskal-Wallis统计量', 'SI正文中', '可保留为SI Table 4', '检查是否还有占位符或缺失值。'],
];
addSheet('04_补充材料编号核对', ['编号','所在文件','内容','当前位置','建议角色','问题/动作'], suppRows, [24,24,44,28,32,64]);

const siRows = [
  ['Supplementary Note 1', '时空掩膜与样本构建', 'dynamic t1边界、排除0类、正负样本定义', '基本已有', '公式和文字需与最终 non-vacant dynamic_t1 口径一致。'],
  ['Supplementary Note 2', '道路网络重建与空间句法变量', 'OSM修复、道路/市政掩膜、时间单调修复、主干筛选、Integration/NAIN/NACH', '已有但需核深度', '方法细节要足够复现，但不要超出实际代码能力。'],
  ['Supplementary Note 3', '多尺度高斯KDE投影', '把路网句法指标投影到10m栅格单元', '已有', '明确是10m×10m，不是30m×30m。'],
  ['Supplementary Note 4', '置换重要性与不用传统SHAP的理由', '基于AUC损失的permutation importance', '已有', '明确不是传统SHAP，而是全局模型无关置换重要性。'],
  ['Supplementary Note 5', 'Built-to-built核验与循环再利用指标', 'built-to-built判据、greenfield track、one-step baseline、remodel-cyclic fraction', '已有', '除非重新放回白盒，否则不要再写white-box。'],
  ['Supplementary Note 6', 'TWFE面板模型设定', '城市固定效应、时期固定效应、解释变量、稳健标准误', '已有', '主文解释要承认当前回归结果整体不显著。'],
  ['质量核验说明', '空地/绿地/市政异常核验', '用地异常和市政转入核验过程', '建议新增Note 7', '老师明确关心市政、空地、绿地问题，SI应保留核验链条。'],
];
addSheet('05_SI方法说明清单', ['SI部分','主题','必须包含','当前状态','动作'], siRows, [28,42,60,24,72]);

const dataRows = [
  ['SD1_Master_79x8', '79城×8时期总表', '城市、省份、区域、规模、时期、样本数、更新数、更新率、扩张面积、更新主导度、AUC/AP', 'Supplementary Data', '必须有'],
  ['SD2_Multimodel_AUC_AP', '三模型AUC/AP矩阵', 'GBDT/RF/L2-LR按城市-时期的AUC和AP，以及时期均值', 'Supplementary Data + Source Data Fig1/EDF1', '必须有'],
  ['SD3_Ablation_M0_M7', '消融实验结果', 'M0-M7 AUC/AP，城市层面或汇总层面', 'Supplementary Data + EDF2', '必须有'],
  ['SD4_Permutation_Importance', '完整置换重要性结果', '特征层面和特征族层面的delta AUC，按时期/城市保存', 'Supplementary Data + Fig3', '必须有'],
  ['SD5_Transitions_All', '完整用地转移矩阵', '所有来源-目标用地类型，不只Top 3', 'Supplementary Data', '必须有'],
  ['SD6_Update_Expansion_Area', '城市-时期更新与扩张面积', 'built-to-built、greenfield、市政扩张/转入等', 'Supplementary Data + Fig2/Fig4/EDF4', '必须有'],
  ['SD7_Regional_Tests', '区域检验完整统计量', 'H值、p值、组内中位数、样本量', 'Supplementary Data + EDF5', '必须有'],
  ['SD8_Syntax_Summary', '空间句法网络指标汇总', '道路长度、连通性、Integration、NAIN、NACH/Choice等', 'Supplementary Data', '对方法可信度很关键'],
  ['SD9_Feature_Dictionary_167', '167维完整特征字典', 'feature_name、family、radius、definition、source script/file', 'Supplementary Data', '必须有'],
  ['SD10_Source_Index', '图表源数据索引', '主文图号/图板与数据表、脚本的对应关系', 'Supplementary Data或README', '强烈建议'],
];
addSheet('06_补充数据工作簿规划', ['Sheet名称','内容','关键字段','用途','优先级'], dataRows, [30,42,68,40,20]);

const issueRows = [
  ['最高', 'Supplementary Table编号冲突', '主文说M0-M7是Supplementary Table 3，SI里M0-M7是Supplementary Table 2。', '统一编号，并同步修改主文、图注和SI。'],
  ['最高', '白盒术语残留', '关键词和built-to-built段落仍有white-box表述。', '如果当前主文已去掉白盒，应改成built-to-built audit / diagnostic audit。'],
  ['高', '摘要可能超词数', '当前英文摘要粗略约158词。', '如Nature Cities Article限制150词，应进一步压缩。'],
  ['高', 'SI中有模板化语言', 'SI仍出现to be populated / template等表达。', '投稿或发老师前必须删除。'],
  ['高', 'Source Data尚未完整打包', '主图和Extended Data图需要对应源数据。', '建立Source_Data.xlsx，每图/图板一个sheet。'],
  ['中', 'Data/Code Availability仍偏泛', '数据和代码可用性声明还需要具体归档方式。', '确定是否开源、归档、或依请求提供。'],
  ['中', '关键参考文献准确性', 'TCULU、Han 2020、Nature Cities城市层级文章需要按老师指定格式核对。', '逐条核对作者、题名、期刊、卷页和年份。'],
];
addSheet('07_下一步行动项', ['优先级','问题','证据','建议动作'], issueRows, [16,36,72,72]);

await fs.mkdir(OUT_DIR, { recursive: true });
const out = await SpreadsheetFile.exportXlsx(wb);
await out.save(OUT);

const sheetList = await wb.inspect({ kind: 'sheet', include: 'id,name', maxChars: 4000 });
console.log(sheetList.ndjson);
console.log(OUT);

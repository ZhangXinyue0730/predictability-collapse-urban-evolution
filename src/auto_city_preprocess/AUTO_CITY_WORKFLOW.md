# 自动化迁移到其他城市的前置流程

本流程把北京/苏州的人工前置处理抽象成两个通用脚本，目标是让程序按城市锚点和行政边界自动生成 pipeline 输入。

## 1. 从全国 TCULU 自动提取城市主体

脚本：

```bash
python auto_city_from_national.py \
  --city-name Suzhou \
  --slug suzhou \
  --lon 120.5853 \
  --lat 31.2989
```

逻辑：

- 根据城市中心经纬度，在全国 2024 候选城市团块标签中自动查找所在或邻近的 `candidate_city_id`。
- 以城市中心为锚点裁取默认 `90km × 90km` 局部窗口。
- 在局部窗口内提取核心城市功能种子：居住、商业、公共服务、工业。
- 对局部种子做连通与合并，保留靠近城市中心的主体团块，避免长三角、珠三角等连续建成区把邻近城市一起纳入。
- 用 2024 主体 mask 回裁 1984、1990、1995、2000、2005、2010、2015、2020、2024 九期土地利用。

主要输出：

```text
city_extract_national/<slug>_candidate/
  <slug>_candidate_2024_raw.tif
  <slug>_core_main_mask_2024.tif
  cropped_years/<year>/landuse_raw.tif
  cropped_years/<year>/mask.tif
  pipeline_city_metadata.json
```

其中 `pipeline_city_metadata.json` 会给出后续 `pipeline/config.py` 需要填写的：

```text
CITY_NAME
H_R
W_R
PX_M
X0_M
Y0_M
```

## 2. 用行政边界自动裁剪城市 OSM 道路

脚本：

```bash
python auto_clip_osm_roads_by_admin.py \
  --city-name 苏州市 \
  --roads-shp jiangsu-260504-free/gis_osm_roads_free_1.shp \
  --admin-shp jiangsu-260504-free/gis_osm_adminareas_a_free_1.shp \
  --output-dir suzhou-260504-free.shp
```

逻辑：

- 从省级 Geofabrik OSM `adminareas` 文件中查找城市行政边界。
- 用该行政边界裁剪省级 `gis_osm_roads_free_1.shp`。
- 输出城市级道路文件，供 `pipeline/01_extract_osm.py` 使用。

主要输出：

```text
<output-dir>/gis_osm_roads_free_1.shp
```

## 3. 后续接入 pipeline

完成上述两步后：

1. 复制一份城市 pipeline 文件夹。
2. 根据 `pipeline_city_metadata.json` 修改新城市 `pipeline/config.py`。
3. 修改桥接脚本中的城市裁剪目录，或继续把桥接脚本参数化。
4. 运行：

```bash
python pipeline/01_extract_osm.py <city_osm_roads_shp>
python pipeline/03.5_bridge_raw_tif_to_pipeline_grid.py
python pipeline/run_all.py
python pipeline/13_update_transition_stats.py
python pipeline/14_period_shap_analysis.py
```

## 4. 当前自动化边界

已经自动化：

- 根据城市经纬度自动定位全国候选团块。
- 自动裁局部窗口。
- 自动提取城市主体 mask。
- 自动回裁九期 TCULU。
- 自动根据行政边界裁剪 OSM 城市道路。
- 自动输出 pipeline 配置元数据。

仍建议人工确认：

- 城市中心经纬度是否合理。
- 行政边界名称是否与 OSM adminareas 中一致。
- 对于超大都市圈，`90km × 90km` 是否需要调整。
- 主体 mask 是否把邻近城市误纳入或漏掉远郊组团。

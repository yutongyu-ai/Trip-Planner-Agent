# Trip Planner Agent（行程规划 Agent）

[English](README.md)

单城市行程规划应用。用户输入目的地和偏好；后端一个 LangGraph agent 动态调用「景点 / 天气 / 路线」三个工具，用基于 embedding 的检索对候选景点排序，合成逐日行程，并在行程违反用户设定的节奏或预算约束时反复重排。

双语（English / 中文）界面面向**出境游**场景——中文用户规划前往国际城市的行程——下方这套国际数据源在这个场景下表现良好。中国大陆目的地会被识别并标记为「低覆盖」；境内数据源方案见 [`docs/adr-002-china-provider.md`](docs/adr-002-china-provider.md)，尚未实现。

## 架构

```
React (Vite + TS)
  Trip Setup ──────────┐
                        ▼
              FastAPI  /trip-plan
                        │
                  LangGraph agent
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
    景点工具          路线工具          天气工具
  (Google Places)     (OSRM)      (OpenWeatherMap)
        │
        ▼
     推荐模块（embedding 检索 + 排序）
```

**LangGraph 在这里的价值在于处理线性流水线做不到的三件事**（不是摆设）：

1. **动态工具选择** —— 根据行程日期/设置决定是否调用某个工具（例如出发日在 5 天之后就完全跳过天气查询，因为 OpenWeatherMap 免费层只覆盖 5 天预报窗口）。
2. **重试 + 优雅降级** —— 候选景点太少时自动放大搜索半径重试（见 `app/tools/places_client.py` 的 `_RADIUS_LADDER_M`）。
3. **有界约束检查循环** —— 行程合成后检查是否违反用户设定的每日节奏/预算；若违反则回到规划节点重排，最多 2 轮。

以上三点都是**写死的控制流**——分支由规则决定，模型从不参与。**`POST /trip-plan/{session_id}/chat`（`app/agent/`）是本项目里唯一一处真正由模型自己决定调用哪个工具、何时收手的地方**：一句自由文本的追加需求（"这个博物馆太贵了，换个便宜点的"、"查一下周三的天气，如果不好就把露天景点换成室内的"）由 `langgraph.prebuilt.create_react_agent` 处理，绑定六个工具（`list_itinerary`、`find_alternative`、`swap_stop`、`remove_stop`、`add_stop`、`check_weather`——见 `app/agent/tools.py`），循环"调用工具 → 观察结果 → 决定下一步"直到满足请求为止。每个工具都只是对生成阶段同一个确定性排程器（`schedule_day_with_lunch`）的薄封装、带校验——模型依然从不自己指定时间或顺序，只负责挑*哪个* place_id，和 `itinerary_node` 的分工原则一致。这个 agent 单独的评估（轨迹长度、非法工具调用率、任务成功率）见 [`eval/agent_chat/`](eval/agent_chat/README.md)，和下面生成流水线自己的 eval 是两回事。已知的范围限制：一次编辑只需要满足排程器自身的可行性（营业时间、通勤时间、午餐），**不会**像生成阶段那样重新对照用户设定的节奏/预算范围检查——所以一次对话编辑有可能让某一天变得比原方案更单薄，而不会告知用户。

同样的三个工具客户端还包了一层 **MCP server**（`app/mcp_server.py`）——对现有 provider `Protocol` 层（`app/tools/providers.py`，见 [ADR-001](docs/adr-001-provider-abstraction.md)）的一层薄声明式封装，让 `search_attractions` / `get_weather_forecast` / `get_distance_matrix` 可以被任意 MCP client（Claude Desktop、另一个 agent）调用，而不只是本项目自己的 LangGraph 节点。`scripts/mcp_client_demo.py` 是一个可运行的演示：通过 stdio 连接，对真实 Google Places API 端到端调用 `search_attractions`。

主生成图也加了一个真正的**人工介入（human-in-the-loop）暂停点**，不只是对话 agent 才有自主性：当 `check_constraints_node` 发现一个 LLM 和有界 refine 循环都自己解决不了的违规——目的地里没有足够便宜的候选点，或者候选池已经用光（`AgentState` 上的 `budget_unavoidable` / `pool_exhausted`）——`human_review_node`（`app/graph/nodes/human_review.py`）会用 LangGraph 的 `interrupt()` 把这次运行暂停下来，而不是悄悄降级成一条提示文字。这种情况下 `POST /trip-plan` 会返回 `{"status": "needs_input", "reason", "violations", "options"}`；`POST /trip-plan/{session_id}/resume` 用来回应它（"accept" 照原样接受，或 "relax_budget" 放宽一档预算再给 refine 循环一次机会）。这需要一个 checkpointer（`MemorySaver`，接在 `app/main.py` 里）才能让暂停状态跨越可能是另一次 HTTP 请求的间隔存活下来——和对话 agent 自己用来存会话记忆的是同一个 LangGraph 原语，用在了不同的问题上。

**LLM 到哪里为止，Python 从哪里开始。** 行程 LLM 只负责*挑选*每天去哪些景点、每个大概待多久，**不**决定到达时间和游玩顺序——那是算术（按营业时间对齐、给通勤留时间、排序以减少在途时间），LLM 做不可靠、重排也纠正不过来。一个确定性排程器（`app/graph/scheduling.py`，一个带时间窗的小 TSP 搜索）来做这件事，一天装不下时丢掉排名最低的景点，并在当天中段留一个午餐时段（固定时长——不推荐餐厅，那是另一个问题）。景点间通勤时间从 OSRM 距离矩阵起步（未覆盖的点对退回直线估算），再修正成现实的门到门时间——OSRM 公共服务器没有路况模型，而游客穿城靠的是公共交通、不是点到点驾车。LLM 会拿到按节奏定的**时间预算**（适中的一天填 ≈ 5.5–7 小时等）和各类别的现实游玩时长，所以行程不会交回半空的一天。

推荐模块是**基于 embedding 的检索式排序**，不是训练出来的模型：候选景点描述和用户偏好文本各自编码（多语 sentence-transformer，使中文偏好文本和中文地点描述落在同一向量空间），按余弦相似度打分，再与归一化评分混合。离线评估在 `eval/offline_recsys/`，用 Yelp Open Dataset 的子集计算 NDCG@k / Precision@k，对比「按评分排序」基线。

Agent 端到端评估在 `eval/agent_e2e/`（[详情 + 完整结果](eval/agent_e2e/README.md)）：合成场景（Places/Weather/Route 用 mock，保留真实 gpt-4o-mini 调用），以 **ablation** 方式跑——把三个「让行程可行」的组件（重排循环、确定性排程器、真实 OSRM 矩阵）逐个独立开关。在 19 个手工构造的场景上，约束满足率从 **0.53（LLM 自排时间、不重排）→ 加确定性排程器后 1.00**，平均 refine 轮数降到 **0.00**。排程器补上、而重排循环永远补不上的那些 case（营业时间强制的顺序、跨街区通勤、9:00 出发撞上 10:00 才开门）都是算术问题、不是措辞问题。（1.00 只代表「构造出的失败模式都覆盖了」，不代表任意真实行程都能过。矩阵变体没有单独增益，因为 eval 用 haversine 模拟了 OSRM 的 table 服务。）搭 eval 的过程还暴露了 4 个单元测试没抓到的问题——包括一个基于「时间窗占用率」的检查，最后发现思路本身是错的、整个删掉了——见其 README。

## 范围

- 每次行程只支持**单个目的地城市**；多城市 / 多段行程是未来工作。
- 支持的是**中国大陆以外的目的地**——那里数据源（Google Places、OpenWeatherMap、OSRM）质量好。中国大陆目的地仍会返回行程，但会在响应的 `notes` 里标记为低覆盖；香港 / 澳门 / 台湾不标记。境内数据源设计见 [`docs/adr-002-china-provider.md`](docs/adr-002-china-provider.md)，它接入的 provider 抽象层见 [`docs/adr-001-provider-abstraction.md`](docs/adr-001-provider-abstraction.md)。

## 当前进度

- [x] 项目脚手架（backend / frontend / docker-compose）
- [x] 三个外部工具客户端（Places / Weather / Route）+ 单元测试
- [x] 推荐模块（基于 embedding 的检索式排序，评分按评论数做贝叶斯收缩，使「3 条五星」压不过「几千条 4.6 星」）+ 离线评估（见 `eval/offline_recsys/`；alpha=0.7 曾使 NDCG@10 较纯评分基线提升约 4.2 倍——该数字用更早的英文单语 embedding 模型测得，换多语模型后待重跑）
- [x] LangGraph agent 核心（`backend/app/graph/`，`POST /trip-plan`）
- [x] Agent 端到端评估（见 `eval/agent_e2e/`，合成场景 + 真实 LLM 调用，重排循环 vs. 单次生成对比）
- [x] 前端页面（Trip Setup / Trip Plan，`frontend/src/`，通过 `POST /trip-plan` 与后端通信）
- [x] 对话式行程编辑 agent（`backend/app/agent/`，`POST /trip-plan/{session_id}/chat`，ReAct + function calling；见 `eval/agent_chat/`）
- [x] 暴露工具客户端的 MCP server（`backend/app/mcp_server.py`，`scripts/mcp_client_demo.py`）
- [x] 主图上的人工介入暂停点 + checkpointing（`backend/app/graph/nodes/human_review.py`，`POST /trip-plan/{session_id}/resume`）

## 本地开发

### 后端

```bash
cd backend
python3 -m venv .venv          # 需要 Python 3.13
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # 核心依赖（API + 工具客户端）
pip install -r requirements-agent.txt    # 追加 LangGraph / 推荐模块依赖（安装体积较大）

cp ../.env.example ../.env  # 填入真实 API key
pytest -q
uvicorn app.main:app --reload
```

### 前端

```bash
cd frontend
npm install
npm run dev
```

### Docker

```bash
docker compose up --build
```

## 获取 API Key

| 服务 | 用途 | 是否需要绑卡 |
|---|---|---|
| [OpenAI](https://platform.openai.com/api-keys) | LangGraph agent LLM（gpt-4o-mini） | 是 |
| [Google Places API (New)](https://console.cloud.google.com/apis/library/places-backend.googleapis.com) | 景点搜索 | 是（有免费额度） |
| [OpenWeatherMap](https://openweathermap.org/api) | 天气预报 | 否（免费层） |
| OSRM 公共 demo server | 路线规划 | 无需 key |

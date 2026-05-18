# GPT-Image-2 接入文档

## 基本信息

| 项目 | 值 |
|------|------|
| 接口地址 | `https://ai.krapi.cn` |
| 请求方式 | POST |
| 认证方式 | `Authorization: Bearer <your_api_key>` |
| 异步查询地址 | `https://ai.krnorth.top/task/{task_id}` |

## 工作流程

本接口为**异步模式**：

```
1. POST 提交生图任务 → 立即返回 task_id（毫秒级响应）
2. GET /task/{task_id} → 轮询直到 status=completed，拿到图片 URL
```

## 支持的接口

```
POST /v1/images/generations    （文生图）
POST /v1/images/edits          （图生图/编辑）
```

---

## 一、文生图

### 接口

```
POST https://ai.krapi.cn/v1/images/generations
```

### 请求示例

```json
{
  "model": "gpt-image-2",
  "prompt": "一只橘猫坐在未来城市的窗边，电影感光影",
  "size": "1024x1024",
  "n": 1,
  "quality": "auto",
  "output_format": "png"
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | 是 | 固定 `gpt-image-2` |
| prompt | string | 是 | 图片描述 |
| size | string | 否 | 图片尺寸，宽x高，必须是 16 的倍数 |
| n | int | 否 | 固定传 `1` |
| quality | string | 否 | `auto` / `high` / `low` |
| output_format | string | 否 | `png` / `jpeg` / `webp` |

### 常用尺寸

```
1024x1024    (1:1)
2048x2048    (1:1 高清)
2048x1152    (16:9)
1152x2048    (9:16)
2048x1536    (4:3)
1536x2048    (3:4)
3840x2160    (16:9 4K)
2160x3840    (9:16 4K)
```

尺寸规则：宽和高都必须是 **16 的倍数**，长边不超过 3840，总像素在 655,360 ~ 8,294,400 之间。

---

## 二、图生图 / 编辑

### 接口

```
POST https://ai.krapi.cn/v1/images/edits
```

### 请求示例（JSON 格式，传图片 URL）

```json
{
  "model": "gpt-image-2",
  "prompt": "把背景换成海边日落，保持主体人物一致",
  "size": "1024x1024",
  "n": 1,
  "images": [
    {
      "image_url": "https://example.com/source.png"
    }
  ]
}
```

### 请求示例（JSON 格式，传 base64）

```json
{
  "model": "gpt-image-2",
  "prompt": "把背景换成森林",
  "size": "1024x1024",
  "n": 1,
  "images": [
    {
      "image_url": "data:image/jpeg;base64,/9j/4AAQ..."
    }
  ]
}
```

### 请求示例（multipart 上传文件）

```bash
curl https://ai.krapi.cn/v1/images/edits \
  -H "Authorization: Bearer your_api_key" \
  -F "model=gpt-image-2" \
  -F "prompt=把背景换成夜晚的纽约街头" \
  -F "size=1024x1024" \
  -F "n=1" \
  -F "image=@/path/to/source.png"
```

### 多张参考图

```bash
curl https://ai.krapi.cn/v1/images/edits \
  -H "Authorization: Bearer your_api_key" \
  -F "model=gpt-image-2" \
  -F "prompt=参考这些图片生成统一风格的新图" \
  -F "n=1" \
  -F "image=@/path/to/source1.png" \
  -F "image=@/path/to/source2.png"
```

### 带蒙版（局部编辑）

```bash
curl https://ai.krapi.cn/v1/images/edits \
  -H "Authorization: Bearer your_api_key" \
  -F "model=gpt-image-2" \
  -F "prompt=只把天空改成晚霞" \
  -F "n=1" \
  -F "image=@/path/to/source.png" \
  -F "mask=@/path/to/mask.png"
```

### 图生图字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | 是 | 固定 `gpt-image-2` |
| prompt | string | 是 | 编辑说明 |
| images | array | JSON 必填 | `[{"image_url": "..."}]`，支持 URL 或 base64 |
| image | file | multipart 必填 | 原图文件 |
| mask | file/object | 否 | 蒙版，用于局部编辑 |
| size | string | 否 | 输出尺寸 |
| n | int | 否 | 固定传 `1` |
| quality | string | 否 | `auto` / `high` / `low` |
| output_format | string | 否 | `png` / `jpeg` / `webp` |

---

## 三、提交响应

提交后立即返回（不等出图）：

```json
{
  "id": "chatcmpl-1747123456789",
  "object": "chat.completion",
  "model": "gpt-image-2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"type\":\"async_task\",\"status\":\"submitted\",\"task_id\":\"xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx\",\"query_url\":\"https://ai.krnorth.top/task/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx\"}"
      },
      "finish_reason": "stop"
    }
  ],
  "status": "submitted",
  "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "query_url": "https://ai.krnorth.top/task/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
}
```

### 关键字段

| 字段 | 说明 |
|------|------|
| task_id | 任务 ID，用于查询结果 |
| query_url | 查询地址，直接 GET 即可 |

---

## 四、查询任务状态

### 请求

```
GET https://ai.krnorth.top/task/{task_id}
```

无需认证头。

### 响应（处理中）

```json
{
  "status": "processing",
  "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "url": null,
  "urls": []
}
```

### 响应（完成）

```json
{
  "status": "completed",
  "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "url": "https://kr-image-xxx.cos.ap-hongkong.myqcloud.com/img_xxx.jpg",
  "urls": ["https://kr-image-xxx.cos.ap-hongkong.myqcloud.com/img_xxx.jpg"],
  "result": {
    "type": "image",
    "url": "https://kr-image-xxx.cos.ap-hongkong.myqcloud.com/img_xxx.jpg"
  }
}
```

### 响应（失败）

```json
{
  "status": "failed",
  "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "error": "Upstream Error 400: ...",
  "detail": "..."
}
```

### 状态值

| status | 含义 |
|--------|------|
| submitted | 已提交，排队中 |
| processing | 正在生成 |
| completed | 完成，url 字段有图片地址 |
| failed | 失败，error 字段有错误信息 |
| not_found_or_expired | 任务不存在或已过期（24小时后清理） |

---

## 五、完整调用示例（Python）

```python
import requests
import time
import json

API_KEY = "your_api_key"
BASE_URL = "https://ai.krapi.cn"

# 1. 提交文生图任务
response = requests.post(
    f"{BASE_URL}/v1/images/generations",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "gpt-image-2",
        "prompt": "一只橘猫坐在窗边看夕阳",
        "size": "2048x1152",
        "n": 1,
        "quality": "high",
    },
    timeout=120,
)

result = response.json()

# 提取 task_id
task_id = result.get("task_id")
query_url = result.get("query_url")

# 如果顶层没有，从 choices 里解析
if not task_id:
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        async_info = json.loads(content)
        task_id = async_info.get("task_id")
        query_url = async_info.get("query_url")
    except:
        pass

print(f"Task ID: {task_id}")
print(f"Query URL: {query_url}")

# 2. 轮询结果（建议间隔 3-5 秒）
for attempt in range(100):
    time.sleep(5)
    poll = requests.get(query_url).json()
    status = poll.get("status", "")
    print(f"[{attempt+1}] status={status}")

    if status == "completed":
        image_url = poll.get("url")
        print(f"✅ 图片地址: {image_url}")
        break
    elif status == "failed":
        print(f"❌ 失败: {poll.get('error')}")
        break
```

---

## 六、完整调用示例（curl）

### 文生图

```bash
# 提交
curl -X POST https://ai.krapi.cn/v1/images/generations \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-image-2","prompt":"画一只橘猫","size":"1024x1024","n":1}'

# 查询（用返回的 task_id 替换）
curl https://ai.krnorth.top/task/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### 图生图

```bash
# 提交（multipart 上传）
curl -X POST https://ai.krapi.cn/v1/images/edits \
  -H "Authorization: Bearer your_api_key" \
  -F "model=gpt-image-2" \
  -F "prompt=把背景换成森林" \
  -F "size=1024x1024" \
  -F "n=1" \
  -F "image=@/path/to/source.png"

# 查询
curl https://ai.krnorth.top/task/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

---

## 七、注意事项

1. **异步模式**：所有请求都是异步的，提交后立即返回 task_id，需要轮询获取结果
2. **图片大小**：参考图建议控制在 10MB 以内
3. **尺寸规则**：宽高必须是 16 的倍数，长宽比不超过 3:1
4. **轮询频率**：建议每 3-5 秒查询一次
5. **任务有效期**：结果保留 24 小时
6. **n 参数**：固定传 1，不支持单次多张
7. **结果格式**：返回的是 COS 托管的图片 URL（JPEG 格式），不是 base64

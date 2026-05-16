# Comfyui-Kr-API

ComfyUI 插件,提供 KR API 中转服务的节点合集,在 ComfyUI 里直接调用 LLM、Gemini 生图、OpenAI 生图、Veo 视频等服务。

## 功能节点

| 节点 | 类型 | 说明 |
|------|------|------|
| KR-语言大模型 | 文本 | 通用 LLM 对话节点,支持多模态输入 |
| KR-Gemini生图 | 图像 | Gemini 文生图 / 图生图,最多 14 张参考图 |
| KR-Gemini异步提交 | 图像 | 异步提交版,适合批量出图。提交后立即返回任务信息 |
| KR-Gemini异步获取 | 图像 | 配合上面的异步提交使用,等待后台任务完成并取回结果 |
| KR-OpenAI生图 | 图像 | OpenAI 兼容图像接口(GPT-Image2 等) |
| KR-Veo3.1视频 | 视频 | Veo 3.1 视频生成 |

## 安装

### 方式一:通过 ComfyUI Manager 安装

在 ComfyUI Manager 里搜索 `Comfyui-Kr-API`,点击安装,重启 ComfyUI 即可。

### 方式二:手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Karenhary/Comfyui-Kr-API.git
cd Comfyui-Kr-API
pip install -r requirements.txt
```

重启 ComfyUI。

## 使用

1. 在 ComfyUI 节点搜索框输入 `KR` 即可看到所有节点,均位于 `KR API中转` 分类下
2. 在节点的 `API密钥` 字段填写 API Key
3. 选择想用的模型预设,或在 `自定义模型` 字段填写模型名

### 异步节点用法

异步节点适合批量任务或耗时较长的请求:

- **KR-Gemini异步提交**:输入参数和同步节点完全一样,运行后立刻返回 `任务信息`(包含 task_id),不会阻塞工作流
- **KR-Gemini异步获取**:在工作流末端放一个,设置最多等待秒数(默认 300 秒),它会等待后台 worker 把所有已提交任务跑完,统一拿回所有图像

提交节点和获取节点不需要连边,只要在同一个工作流里出现即可。

## 系统要求

- ComfyUI(Python 3.10+)
- 自动满足:`numpy`、`Pillow`、`torch`(ComfyUI 已自带)
- 需要安装:`requests`(`pip install -r requirements.txt`)

## 许可证

[MIT License](./LICENSE)

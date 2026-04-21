# LLM Vision-Language Service — API Reference & Deployment Guide

> **Model**: `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`  
> **Engine**: vLLM 0.19.0 (OpenAI-compatible API)  
> **Hardware**: NVIDIA GeForce RTX 3090 (24 GB VRAM)  
> **Quantization**: INT4 AWQ (Marlin kernel) — ~19.5 GB model weights  

---

## Table of Contents

1. [Connection Info](#1-connection-info)
2. [API Endpoints](#2-api-endpoints)LLM_SERVICE.md
3. [Text-Only Chat Completion](#3-text-only-chat-completion)
4. [Vision: Sending Images](#4-vision-sending-images)
   - [Base64 Encoded Image](#41-base64-encoded-image)
   - [Image URL](#42-image-url)
   - [Multiple Images](#43-multiple-images)
5. [Streaming Responses](#5-streaming-responses)
6. [Thinking Mode (Chain-of-Thought)](#6-thinking-mode-chain-of-thought)
7. [Parameters Reference](#7-parameters-reference)
8. [Error Handling](#8-error-handling)
9. [Health Checking](#9-health-checking)
10. [Code Examples](#10-code-examples)
    - [Python (requests)](#101-python-requests)LLM_SERVICE.md
    - [Python (openai SDK)](#102-python-openai-sdk)
    - [Node.js (fetch)](#103-nodejs-fetch)
    - [Node.js (http)](#104-nodejs-http)
    - [cURL](#105-curl)
11. [Infrastructure & Deployment](#11-infrastructure--deployment)
12. [Runtime Configuration](#12-runtime-configuration)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Connection Info

| Property        | Value                                        |
|-----------------|----------------------------------------------|
| **Host**        | `192.168.86.48`                              |
| **Port**        | `8001`                                       |
| **Base URL**    | `http://192.168.86.48:8001`                  |
| **API Style**   | OpenAI-compatible (vLLM built-in server)     |
| **Auth**        | None (LAN-only, no API key required)         |
| **Protocol**    | HTTP (not HTTPS)                             |
| **Model ID**    | `Qwen/Qwen2.5-VL-32B-Instruct-AWQ`         |

The service is exposed on the LAN only. No API key or Bearer token is needed. If you're connecting from outside the LAN, you'll need to set up port forwarding or a tunnel.

**SSH access to the GPU server** (for management/deployment):
| Property         | Value                          |
|------------------|--------------------------------|
| **User**         | `darkmatter2222`               |
| **Host**         | `192.168.86.48`                |
| **Auth**         | SSH key (`~/.ssh/id_rsa`)      |
| **No password**  | Key-based auth only            |

---

## 2. API Endpoints

All endpoints follow the OpenAI API format.

| Method | Path                         | Description                          |
|--------|------------------------------|--------------------------------------|
| GET    | `/health`                    | Health check (returns 200 if ready)  |
| GET    | `/v1/models`                 | List available models                |
| POST   | `/v1/chat/completions`       | Chat completion (text + images)      |
| POST   | `/v1/completions`            | Text completion (prompt-based)       |
| POST   | `/v1/chat/completions/render`| Render chat template without executing |
| POST   | `/v1/completions/render`     | Render completion template           |

The primary endpoint you'll use is **`POST /v1/chat/completions`**.

---

## 3. Text-Only Chat Completion

### Request

```http
POST http://192.168.86.48:8001/v1/chat/completions
Content-Type: application/json
```

```json
{
  "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the capital of France?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

### Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1776466400,
  "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 8,
    "total_tokens": 32
  }
}
```

### Key Notes

- **`chat_template_kwargs.enable_thinking`**: Set to `false` to get direct answers. Set to `true` to enable chain-of-thought reasoning (see [Section 6](#6-thinking-mode-chain-of-thought)).
- The `model` field is optional — the server only hosts one model.
- Multi-turn conversations are supported by including the full message history in the `messages` array.

---

## 4. Vision: Sending Images

This is a **vision-language model** (VL). It can analyze images alongside text prompts. Images are sent as part of the `content` field using the OpenAI multimodal message format.

The `content` field becomes an **array** of content parts instead of a plain string. Each part has a `type`:

| Type         | Description                     |
|--------------|---------------------------------|
| `text`       | Text content                    |
| `image_url`  | An image (base64 or URL)        |

### 4.1 Base64 Encoded Image

The most reliable method. Encode the image as base64 and embed it in a data URI.

**Supported formats**: JPEG, PNG, GIF, WebP, BMP, TIFF

```json
{
  "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
          }
        },
        {
          "type": "text",
          "text": "Describe what you see in this image in detail."
        }
      ]
    }
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

**Data URI format**: `data:image/<format>;base64,<base64_data>`

| Image Format | MIME Type       | Data URI Prefix                    |
|-------------|-----------------|-------------------------------------|
| JPEG        | `image/jpeg`    | `data:image/jpeg;base64,`          |
| PNG         | `image/png`     | `data:image/png;base64,`           |
| GIF         | `image/gif`     | `data:image/gif;base64,`           |
| WebP        | `image/webp`    | `data:image/webp;base64,`          |

### 4.2 Image URL

You can also pass an HTTP/HTTPS URL. The server will fetch the image at inference time.

> **Note**: The server must be able to reach the URL from its network (LAN or internet).

```json
{
  "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {
            "url": "https://example.com/photo.jpg"
          }
        },
        {
          "type": "text",
          "text": "What objects are in this photograph?"
        }
      ]
    }
  ],
  "max_tokens": 512,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

### 4.3 Multiple Images

You can send multiple images in a single message by adding multiple `image_url` content parts. The current server configuration supports up to **4 images per request** as a practical limit given VRAM constraints.

```json
{
  "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": { "url": "data:image/png;base64,iVBOR..." }
        },
        {
          "type": "image_url",
          "image_url": { "url": "data:image/png;base64,iVBOR..." }
        },
        {
          "type": "text",
          "text": "Compare these two images. What are the differences?"
        }
      ]
    }
  ],
  "max_tokens": 1024,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

### Image Size Considerations

- Large images are resized internally by the vision encoder. You don't need to pre-resize.
- However, larger images consume more of the 2,048-token context window (images are tokenized into visual tokens).
- For best results, keep images under ~1920×1080. Extremely large images may cause out-of-memory errors.
- The vision encoder budget is 16,384 tokens for image encoding.

---

## 5. Streaming Responses

For real-time token-by-token output, set `"stream": true`. The response uses **Server-Sent Events (SSE)**.

### Request

```json
{
  "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
  "messages": [
    { "role": "user", "content": "Write a haiku about coding." }
  ],
  "stream": true,
  "stream_options": { "include_usage": true },
  "max_tokens": 256,
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

### Response (SSE stream)

```
data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Lines"},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" of"},"finish_reason":null}]}

data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" code"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-xyz","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":14,"completion_tokens":17,"total_tokens":31}}

data: [DONE]
```

### Parsing SSE

1. Each line starts with `data: `.
2. Parse the JSON payload after `data: `.
3. Extract `choices[0].delta.content` for each chunk — concatenate to build the full response.
4. `data: [DONE]` signals the end of the stream.
5. If `stream_options.include_usage` is `true`, the final chunk before `[DONE]` includes a `usage` object.

---

## 6. Thinking Mode (Chain-of-Thought)

Qwen2.5-VL supports a "thinking" mode where the model produces internal reasoning before answering. This is controlled by the `chat_template_kwargs.enable_thinking` parameter.

### Thinking Disabled (Default for this deployment)

```json
{
  "chat_template_kwargs": { "enable_thinking": false }
}
```

The model responds directly. Best for structured outputs, JSON responses, and fast answers.

### Thinking Enabled

```json
{
  "chat_template_kwargs": { "enable_thinking": true },
  "temperature": 0.6,
  "top_p": 0.95
}
```

The model produces `<think>...</think>` blocks before its final answer. In streaming mode, thinking tokens arrive as `reasoning_content` in the delta:

```json
{
  "choices": [{
    "delta": {
      "reasoning_content": "Let me analyze this step by step..."
    }
  }]
}
```

The final answer follows as regular `content`.

> **Tip**: When thinking is enabled, use `temperature: 0.6` and `top_p: 0.95` for best results (per Qwen's recommendations). With thinking disabled, `temperature: 0.7` and `top_p: 0.8` work well.

> **Warning**: Thinking mode uses more tokens. With `max_model_len: 2048`, you'll want to be economical with input tokens to leave room for both thinking and the response.

---

## 7. Parameters Reference

### Request Body (`POST /v1/chat/completions`)

| Parameter                | Type      | Default  | Description                                           |
|--------------------------|-----------|----------|-------------------------------------------------------|
| `model`                  | string    | (server) | Model ID. Optional — server only has one model.       |
| `messages`               | array     | required | Array of message objects (see below).                 |
| `temperature`            | float     | 0.7      | Sampling temperature. 0 = deterministic, 2 = max random. |
| `max_tokens`             | int       | 512      | Max tokens to generate in the response.               |
| `top_p`                  | float     | 0.9      | Nucleus sampling cutoff.                              |
| `top_k`                  | int       | -1       | Top-K sampling. -1 = disabled.                        |
| `stop`                   | string[]  | null     | Stop sequences — generation halts when any is produced.|
| `stream`                 | bool      | false    | Enable SSE streaming.                                 |
| `stream_options`         | object    | null     | `{ "include_usage": true }` to get token counts with stream. |
| `presence_penalty`       | float     | 0.0      | Penalize tokens already present. Range: -2.0 to 2.0. |
| `frequency_penalty`      | float     | 0.0      | Penalize frequent tokens. Range: -2.0 to 2.0.        |
| `chat_template_kwargs`   | object    | {}       | `{ "enable_thinking": false }` to disable CoT.       |

### Message Object

| Field     | Type          | Description                                               |
|-----------|---------------|-----------------------------------------------------------|
| `role`    | string        | One of: `system`, `user`, `assistant`                     |
| `content` | string/array  | Text string, or array of content parts for multimodal.    |

### Content Part (Multimodal)

| Field       | Type   | Description                                       |
|-------------|--------|---------------------------------------------------|
| `type`      | string | `"text"` or `"image_url"`                         |
| `text`      | string | The text content (when `type` = `"text"`).        |
| `image_url` | object | `{ "url": "..." }` (when `type` = `"image_url"`). |

### Response Object

| Field                    | Type   | Description                                  |
|--------------------------|--------|----------------------------------------------|
| `id`                     | string | Unique completion ID.                        |
| `object`                 | string | `"chat.completion"`                          |
| `choices`                | array  | Array of choice objects.                     |
| `choices[].message.role` | string | Always `"assistant"`.                        |
| `choices[].message.content` | string | The generated text.                       |
| `choices[].finish_reason`| string | `"stop"` = natural end, `"length"` = hit max_tokens. |
| `usage.prompt_tokens`    | int    | Tokens in the input.                         |
| `usage.completion_tokens`| int    | Tokens generated.                            |
| `usage.total_tokens`     | int    | Sum of prompt + completion.                  |

---

## 8. Error Handling

### HTTP Status Codes

| Code | Meaning                         | Typical Cause                              |
|------|----------------------------------|--------------------------------------------|
| 200  | Success                          | Request completed.                         |
| 400  | Bad Request                      | Malformed JSON, invalid parameters.        |
| 422  | Unprocessable Entity             | Schema validation failure.                 |
| 500  | Internal Server Error            | Model inference error, OOM.                |
| 503  | Service Unavailable              | Model still loading / not ready.           |

### Error Response Format

```json
{
  "object": "error",
  "message": "Error description here.",
  "type": "invalid_request_error",
  "code": 400
}
```

### Common Errors

| Error Message                                | Cause & Fix                                                                                    |
|----------------------------------------------|------------------------------------------------------------------------------------------------|
| `This model's maximum context length is 2048 tokens` | Input + max_tokens exceeds 2048. Shorten input or reduce `max_tokens`.          |
| `Connection refused`                         | Container not running. SSH in and check `docker ps`.                                           |
| `Request timed out`                          | Inference taking too long. Reduce `max_tokens` or simplify the prompt.                         |
| `CUDA out of memory`                         | Too many concurrent requests or too many/large images. Reduce load or image count.             |

---

## 9. Health Checking

### Simple Health Check

```http
GET http://192.168.86.48:8001/health
```

Returns `200 OK` when the model is loaded and ready to serve requests.

Returns `503` or connection refused if the model is still loading or the container is down.

### Model List

```http
GET http://192.168.86.48:8001/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
      "object": "model",
      "created": 1776466339,
      "owned_by": "vllm",
      "root": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
      "max_model_len": 2048
    }
  ]
}
```

### Recommended Health Check Pattern

```python
import requests

def is_llm_ready(host="192.168.86.48", port=8001, timeout=5):
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=timeout)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False
```

---

## 10. Code Examples

### 10.1 Python (requests)

#### Text-Only

```python
import requests

response = requests.post(
    "http://192.168.86.48:8001/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in 3 sentences."}
        ],
        "temperature": 0.7,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False}
    },
    timeout=60
)

data = response.json()
print(data["choices"][0]["message"]["content"])
```

#### With Image (base64)

```python
import requests
import base64

# Read and encode image
with open("photo.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "http://192.168.86.48:8001/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in detail."
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False}
    },
    timeout=120
)

data = response.json()
print(data["choices"][0]["message"]["content"])
```

### 10.2 Python (openai SDK)

The vLLM server is fully compatible with the official OpenAI Python SDK.

```bash
pip install openai
```

#### Text-Only

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.86.48:8001/v1",
    api_key="not-needed"  # Required by the SDK but not validated
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 42 * 37?"}
    ],
    temperature=0.7,
    max_tokens=256,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)

print(response.choices[0].message.content)
```

#### With Image (base64)

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.86.48:8001/v1",
    api_key="not-needed"
)

with open("screenshot.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                },
                {
                    "type": "text",
                    "text": "What does this screenshot show?"
                }
            ]
        }
    ],
    max_tokens=1024,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)

print(response.choices[0].message.content)
```

#### Streaming

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.86.48:8001/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    messages=[
        {"role": "user", "content": "Tell me a short story."}
    ],
    stream=True,
    max_tokens=512,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

### 10.3 Node.js (fetch)

#### Text-Only

```javascript
const response = await fetch("http://192.168.86.48:8001/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    messages: [
      { role: "system", content: "You are a helpful assistant." },
      { role: "user", content: "Hello!" }
    ],
    temperature: 0.7,
    max_tokens: 256,
    chat_template_kwargs: { enable_thinking: false }
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

#### With Image (base64)

```javascript
import { readFile } from "fs/promises";

const imageBuffer = await readFile("photo.jpg");
const base64Image = imageBuffer.toString("base64");

const response = await fetch("http://192.168.86.48:8001/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image_url",
            image_url: { url: `data:image/jpeg;base64,${base64Image}` }
          },
          {
            type: "text",
            text: "What is in this image?"
          }
        ]
      }
    ],
    max_tokens: 1024,
    chat_template_kwargs: { enable_thinking: false }
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

### 10.4 Node.js (http — no dependencies)

```javascript
const http = require("http");

function chatCompletion(messages, options = {}) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({
      model: "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
      messages,
      temperature: options.temperature ?? 0.7,
      max_tokens: options.max_tokens ?? 512,
      chat_template_kwargs: { enable_thinking: false },
    });

    const req = http.request({
      hostname: "192.168.86.48",
      port: 8001,
      path: "/v1/chat/completions",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
      },
      timeout: 60000,
    }, (res) => {
      let data = "";
      res.on("data", (chunk) => data += chunk);
      res.on("end", () => {
        try {
          const json = JSON.parse(data);
          resolve(json.choices[0].message.content);
        } catch (e) {
          reject(new Error(`Parse error: ${data.slice(0, 200)}`));
        }
      });
    });

    req.on("error", reject);
    req.on("timeout", () => { req.destroy(); reject(new Error("Timeout")); });
    req.write(body);
    req.end();
  });
}

// Usage
const answer = await chatCompletion([
  { role: "user", content: "What is 2+2?" }
]);
console.log(answer);
```

### 10.5 cURL

#### Text-Only

```bash
curl http://192.168.86.48:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the meaning of life?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

#### With Image (base64 via file)

```bash
# Encode image to base64
IMG_B64=$(base64 -w0 photo.jpg)

curl http://192.168.86.48:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"Qwen/Qwen2.5-VL-32B-Instruct-AWQ\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,${IMG_B64}\"}},
        {\"type\": \"text\", \"text\": \"Describe this image.\"}
      ]
    }],
    \"max_tokens\": 1024,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }"
```

#### Health Check

```bash
curl http://192.168.86.48:8001/health
```

#### List Models

```bash
curl http://192.168.86.48:8001/v1/models | python3 -m json.tool
```

---

## 11. Infrastructure & Deployment

### Server Hardware

| Component     | Specification                                  |
|---------------|------------------------------------------------|
| **GPU**       | NVIDIA GeForce RTX 3090 (24 GB GDDR6X)        |
| **Compute**   | Compute Capability 8.6 (Ampere)                |
| **OS**        | Ubuntu 24.04 LTS                               |
| **Driver**    | NVIDIA 580.126.09                              |
| **Docker**    | 29.1.0 with NVIDIA Container Toolkit           |
| **CUDA**      | 12.4 (container base image)                    |

### Docker Container

| Property          | Value                                                  |
|-------------------|--------------------------------------------------------|
| **Image**         | `atomic-family-llm:latest`                             |
| **Base**          | `nvidia/cuda:12.4.0-devel-ubuntu22.04`                 |
| **Container**     | `atomic-family-llm`                                    |
| **Port mapping**  | `8001` (host) → `8000` (container)                     |
| **GPU access**    | `--gpus all` (NVIDIA runtime)                          |
| **Restart**       | `unless-stopped`                                       |
| **Health check**  | `GET /health` every 30s, 600s start period             |

### Volumes

| Host Path                                        | Container Path                    | Purpose                         |
|--------------------------------------------------|-----------------------------------|---------------------------------|
| `/home/darkmatter2222/.cache/huggingface`        | `/root/.cache/huggingface`        | Model weights (persistent)      |
| `/home/darkmatter2222/.cache/vllm`               | `/root/.cache/vllm`               | Compiled CUDA graphs (cached)   |

Model weights are **not baked into the Docker image**. They are downloaded on first run and cached persistently in the HuggingFace cache volume. Subsequent starts load from cache (~3.5 seconds).

### Deployment Commands

From the project root (requires bash — Git Bash on Windows or WSL):

```bash
# Full deploy: copy files, build image, start container
bash llm/deploy.sh deploy

# Individual steps
bash llm/deploy.sh build    # Copy files to server & build Docker image
bash llm/deploy.sh start    # Start (or restart) the container
bash llm/deploy.sh stop     # Stop and remove the container
bash llm/deploy.sh status   # Check container status + GPU utilization
bash llm/deploy.sh logs     # Tail container logs (last 50 lines)
```

### Manual Docker Run (current production command)

```bash
docker run -d \
  --name atomic-family-llm \
  --gpus all \
  --restart unless-stopped \
  -p 8001:8000 \
  -v /home/darkmatter2222/.cache/huggingface:/root/.cache/huggingface \
  -v /home/darkmatter2222/.cache/vllm:/root/.cache/vllm \
  -e LLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
  -e LLM_PORT=8000 \
  atomic-family-llm:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-32B-Instruct-AWQ \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --dtype auto \
    --enable-prefix-caching \
    --max-num-seqs 8
```

---

## 12. Runtime Configuration

These are the **actual runtime parameters** of the currently deployed server.

### vLLM Server Arguments

| Parameter                   | Value    | Description                                        |
|-----------------------------|----------|----------------------------------------------------|
| `--model`                   | `Qwen/Qwen2.5-VL-32B-Instruct-AWQ` | Model identifier                |
| `--host`                    | `0.0.0.0`| Bind to all interfaces                             |
| `--port`                    | `8000`   | Internal container port                            |
| `--max-model-len`           | `2048`   | Maximum context window (tokens)                    |
| `--gpu-memory-utilization`  | `0.95`   | Fraction of GPU memory to use (~23.3 GB / 24.6 GB) |
| `--trust-remote-code`       | true     | Required for Qwen models                           |
| `--dtype`                   | `auto`   | Auto-detects bfloat16 for this model               |
| `--enable-prefix-caching`   | true     | Reuses KV cache for shared prompt prefixes         |
| `--max-num-seqs`            | `8`      | Max concurrent sequences (batched inference)       |

### Memory Budget Breakdown

| Component              | Memory         |
|-----------------------|----------------|
| **Model weights**      | 19.54 GiB      |
| **KV cache**           | ~0.51 GiB      |
| **KV cache capacity**  | 2,080 tokens   |
| **CUDA graphs + overhead** | ~3.5 GiB  |
| **Total allocated**    | ~23.3 GiB / 24 GiB |

### Quantization

| Property          | Value                                |
|-------------------|--------------------------------------|
| **Method**        | AWQ (Activation-aware Weight Quantization) |
| **Precision**     | INT4 (4-bit weights)                 |
| **Kernel**        | Marlin (optimized CUDA kernel)       |
| **Model dtype**   | bfloat16 (activations)               |
| **Weight size**   | ~18 GiB (quantized)                  |
| **Original size** | ~64 GiB (FP16 unquantized)          |

### Model Details

| Property             | Value                                    |
|----------------------|------------------------------------------|
| **Full name**        | Qwen/Qwen2.5-VL-32B-Instruct-AWQ        |
| **Architecture**     | Qwen2_5_VLForConditionalGeneration       |
| **Parameters**       | 32 billion (quantized to INT4)           |
| **Type**             | Vision-Language (multimodal)             |
| **Text support**     | Yes — full chat/instruction following    |
| **Image support**    | Yes — single and multi-image             |
| **Video support**    | Yes — via frame extraction (limited by memory) |
| **Languages**        | Multilingual (English, Chinese, and more) |
| **Context window**   | 2,048 tokens (limited by VRAM; native support up to 32K) |
| **Encoder budget**   | 16,384 tokens for vision encoding        |
| **License**          | Apache 2.0                               |

### Startup Timeline

| Phase                     | Duration     |
|---------------------------|-------------|
| Container start + init    | ~10s        |
| Model weight loading      | ~3.5s       |
| Encoder cache profiling   | ~21s        |
| torch.compile (cached)    | ~33s        |
| KV cache allocation       | ~2s         |
| Route registration        | <1s         |
| **Total cold start**      | **~70s**    |

> Second startups with cached CUDA graphs (`/root/.cache/vllm` volume) are faster.

---

## 13. Troubleshooting

### Container won't start

```bash
# Check container logs
ssh darkmatter2222@192.168.86.48 "docker logs atomic-family-llm 2>&1 | tail -30"

# Check if port 8001 is in use
ssh darkmatter2222@192.168.86.48 "ss -tlnp | grep 8001"

# Check GPU memory
ssh darkmatter2222@192.168.86.48 "nvidia-smi"
```

### OOM (Out of Memory) errors

The RTX 3090 has 24 GB. The model uses ~19.5 GB, leaving only ~4.5 GB for KV cache, CUDA graphs, and overhead. If you hit OOM:

1. Reduce `--max-model-len` (current: 2048, can go as low as 512)
2. Reduce `--max-num-seqs` (current: 8, lower = fewer concurrent requests)
3. Add `--enforce-eager` to skip CUDA graph capture (saves ~1 GB but slower inference)
4. Kill other GPU processes: `nvidia-smi` to find PIDs, then `kill <pid>`

### Model not responding

```bash
# Check container is running
docker ps -f name=atomic-family-llm

# Check health
curl http://192.168.86.48:8001/health

# Restart container
docker restart atomic-family-llm

# Full restart (stop + start fresh)
docker rm -f atomic-family-llm
# Then run the docker run command from section 11
```

### Slow responses

- **First request after startup**: Expect 5–10 seconds (CUDA warmup). Subsequent requests are faster.
- **Large images**: Resize to ≤1920×1080 before sending.
- **Long prompts**: The 2048 context window is tight. Keep prompts concise.
- **Too many concurrent requests**: `max_num_seqs=8` limits batching. Queue excess requests.

### Testing the connection

```bash
# Quick smoke test
curl -s http://192.168.86.48:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello"}],"max_tokens":16}' \
  | python3 -m json.tool
```

---

## Environment Variables Summary

For any application connecting to this service, configure these variables:

```env
LLM_HOST=192.168.86.48
LLM_PORT=8001
LLM_MODEL=Qwen/Qwen2.5-VL-32B-Instruct-AWQ
LLM_ENABLED=true
```

The base URL is: `http://${LLM_HOST}:${LLM_PORT}`

No API key is needed. All requests are unauthenticated.

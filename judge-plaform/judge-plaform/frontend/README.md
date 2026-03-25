# 前端部署说明（Nginx）

## 1) 静态文件

将 `frontend/index.html` 拷贝到 Nginx 静态目录，例如：

- Linux 常见路径：`/usr/share/nginx/html/index.html`
- Windows 常见路径：`nginx/html/index.html`

## 2) 推荐 Nginx 配置

如果页面和后端不在同一端口，建议通过 Nginx 反向代理后端接口，前端继续调用 `/judge/submit`，避免跨域问题：

```nginx
server {
    listen 80;
    server_name localhost;

    root html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /judge/ {
        proxy_pass http://127.0.0.1:8080/judge/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

> 如果你不做反向代理，也可以在页面顶部把接口地址改成完整地址，例如：`http://127.0.0.1:8080/judge/submit`。

## 3) 页面能力

- 填写待审核文本、审核原因（限制 50 字）
- 从文本中框选片段并打违规类型标签
- 违规类型固定：`normal | abuse | violence | porn | politics | other`
- 当前后端通过 `ThreadLocal` 固定写入审核员ID：`1001`（硬编码）
- 后端自动用雪花算法生成 `case_id`（Long）
- 提交时将证据文本数组写入 `evidence`（JSON 数组）
- 请求路径：`POST /judge/submit`

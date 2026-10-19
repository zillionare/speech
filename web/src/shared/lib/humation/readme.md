来源：Nexus 内置 Humation，https://github.com/humation-labs/humation 固定提交 f47bb3dee04f21e377b6114141737280b8daa68b，素材版本 1.0.1。

保留上游源码、素材、注释及 MIT LICENSE；create-avatar.ts 额外导出 fnv1a，nexus-parts.ts 为 Nexus 自有配件。头像按资源 ID 稳定生成，不随名称变化。

本地调整：assets.ts 中 SVG 的路径坐标统一保留 1 位小数并压缩空白（画布 80×80，显示误差远小于 1 像素），用于减小按需加载的头像包体积。

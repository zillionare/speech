# Astryx 弹层兼容补丁

`@astryxdesign/core@0.6.1.patch` 对应 issue #6：在缺少 CSS 锚点定位的 WebKit/WebView 中，由共享 `useLayer` 接入 `@floating-ui/dom` 计算菜单位置；保留原组件的样式、焦点管理和键盘交互。自定义坐标模式不受影响。

定位使用 Floating UI 的 `computePosition`、`offset/flip/shift/size` 和 `autoUpdate`，不自行实现坐标计算或滚动监听。`packageExtensions` 为被补丁修改的 Astryx 显式补充依赖，确保 pnpm 隔离安装也能解析。

参考：[Floating UI](https://floating-ui.com/docs/computeposition)、[Radix Popper](https://github.com/radix-ui/primitives/blob/main/packages/react/popper/src/popper.tsx)、[Headless UI](https://github.com/tailwindlabs/headlessui/blob/main/packages/@headlessui-react/src/internal/floating.tsx)。

补丁同时维护 `src/Layer` 和实际发布的 `dist/Layer`。通过 `pnpm patch` / `pnpm patch-commit` 更新，安装依赖时由 pnpm 自动应用。升级 Astryx 后，先检查上游是否已修复，再决定移除或迁移补丁。

验证：`pnpm --dir web run test legacy-selectors.spec.ts`。测试覆盖缺少锚点定位、菜单避让、滚动、缩放、弹窗、键盘和无 Popover 时的原生兜底。

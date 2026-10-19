// 本机加载往往只要几百毫秒，动画还没淡入就被移除；就绪时至少显示到这个时刻（从页面开始加载算起）。
const minimumVisibleMs = 1200;
let pending = 0;

/**
 * 移除 index.html 中的启动动画：先淡出，淡出结束后删除节点；可重复调用。
 * immediate 用于出错的情况，立即淡出（包括正在等待最短显示时间时），不让动画挡住错误提示。
 */
export function dismissSplash(immediate = false) {
  const splash = document.getElementById('splash');
  if (!splash || splash.dataset.leaving) return;
  const leave = () => {
    window.clearTimeout(pending);
    splash.dataset.leaving = 'true';
    const remove = () => splash.remove();
    splash.addEventListener('transitionend', remove, { once: true });
    // 减少动态效果或页面在后台时可能收不到 transitionend，到时直接删除。
    window.setTimeout(remove, 800);
  };
  const wait = immediate ? 0 : minimumVisibleMs - performance.now();
  if (wait <= 0) leave();
  else if (!pending) pending = window.setTimeout(leave, wait);
}

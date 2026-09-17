//! 运气分 SVG 出图（零 Python 依赖、零第三方绘图依赖）
//!
//! - [`svg`]：极简 SVG 构建器（线 / 折线 / 矩形 / 圆 / 文本，字符串拼接出合法 SVG 1.1）
//! - [`luck_trend`]：单局趋势图（期望评分 / 运气分 / 运气波动 3 子图，样式对齐
//!   `scripts/plot_luck_trend.py`），供局末自动出图与离线/在线复盘入口复用
//!
//! 输出为自包含单文件 `luck_trend.svg`（不内嵌字形，中文走渲染器字体回退），
//! 不产出 PNG/JPG、不引入 `resvg` 等栅格化依赖。

pub mod luck_trend;
pub mod svg;
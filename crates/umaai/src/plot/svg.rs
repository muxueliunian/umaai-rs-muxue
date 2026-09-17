//! 极简 SVG 1.1 构建器（零第三方依赖）
//!
//! 只提供绘图所需的元素（`rect` / `line` / `polyline` / `circle` / `text`），
//! 内部以字符串拼接维护元素列表，最终拼成完整 `<svg>` 文档。文本统一做 XML
//! 转义；中文字形交给渲染器字体回退（不内嵌字形）。

/// 一个待输出的 SVG 文档
pub struct Svg {
    w: f64,
    h: f64,
    parts: Vec<String>,
}

/// XML 转义（`&` / `<` / `>` / 引号）
fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

impl Svg {
    /// 新建画布（宽高 = viewBox 尺寸）
    pub fn new(w: f64, h: f64) -> Self {
        Self { w, h, parts: Vec::new() }
    }

    /// 追加一段原始 SVG 片段（构建器自身方法已足够时无需外部调用）
    pub fn push(&mut self, raw: impl Into<String>) {
        self.parts.push(raw.into());
    }

    /// 矩形（`fill` 支持十六进制 / 命名色；`opacity` 0..1）
    pub fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: &str, opacity: f64) {
        self.push(format!(
            "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" fill=\"{fill}\" fill-opacity=\"{opacity}\"/>"
        ));
    }

    /// 线段
    pub fn line(&mut self, x1: f64, y1: f64, x2: f64, y2: f64, stroke: &str, width: f64) {
        self.push(format!(
            "<line x1=\"{x1:.1}\" y1=\"{y1:.1}\" x2=\"{x2:.1}\" y2=\"{y2:.1}\" stroke=\"{stroke}\" stroke-width=\"{width:.1}\"/>"
        ));
    }

    /// 描边矩形（子图边框 / 坐标轴框；`fill=none`）
    pub fn frame(&mut self, x: f64, y: f64, w: f64, h: f64, stroke: &str, width: f64) {
        self.push(format!(
            "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{w:.1}\" height=\"{h:.1}\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"{width:.1}\"/>"
        ));
    }

    /// 折线（数据点坐标序列；`dashed` 时画虚线）
    pub fn polyline(&mut self, pts: &[(f64, f64)], stroke: &str, width: f64, dashed: bool) {
        if pts.len() < 2 {
            return;
        }
        let points = pts
            .iter()
            .map(|(x, y)| format!("{x:.1},{y:.1}"))
            .collect::<Vec<_>>()
            .join(" ");
        let dash = if dashed { " stroke-dasharray=\"6 4\"" } else { "" };
        self.push(format!(
            "<polyline points=\"{points}\" fill=\"none\" stroke=\"{stroke}\" stroke-width=\"{width:.1}\"{dash}/>"
        ));
    }

    /// 圆点（数据点标记）
    pub fn circle(&mut self, x: f64, y: f64, r: f64, fill: &str, opacity: f64) {
        self.push(format!(
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"{r:.1}\" fill=\"{fill}\" fill-opacity=\"{opacity}\"/>"
        ));
    }

    /// 文本（`anchor` ∈ start / middle / end）
    pub fn text(&mut self, x: f64, y: f64, s: &str, size: f64, anchor: &str) {
        self.push(format!(
            "<text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"{size:.0}\" text-anchor=\"{anchor}\">{}</text>",
            esc(s)
        ));
    }

    /// 旋转文本（纵轴标题用；`deg` 为绕 `(x, y)` 的旋转角，SVG 中 -90 = 自下而上竖排）
    pub fn text_rot(&mut self, x: f64, y: f64, s: &str, size: f64, anchor: &str, deg: f64) {
        self.push(format!(
            "<text x=\"{x:.1}\" y=\"{y:.1}\" font-size=\"{size:.0}\" text-anchor=\"{anchor}\" transform=\"rotate({deg:.0} {x:.1} {y:.1})\">{}</text>",
            esc(s)
        ));
    }

    /// 完整文档（含 xml 头与字体回退栈）
    pub fn render(&self) -> String {
        let mut s = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        s.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {:.0} {:.0}\" font-family=\"Microsoft YaHei, PingFang SC, Noto Sans CJK SC, sans-serif\">\n",
            self.w, self.h
        ));
        s.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n");
        for p in &self.parts {
            s.push_str(p);
            s.push('\n');
        }
        s.push_str("</svg>\n");
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// XML 转义：`&` / `<` / `>` / 引号
    #[test]
    fn test_svg_escape() {
        assert_eq!(esc("A&B"), "A&amp;B");
        assert_eq!(esc("<b>"), "&lt;b&gt;");
        assert_eq!(esc("a\"b"), "a&quot;b");
        assert_eq!(esc("训练/速"), "训练/速");
    }

    /// 空文档也能产出合法结构（xml 头 + svg 根 + 白底）
    #[test]
    fn test_svg_render_minimal() {
        let svg = Svg::new(100.0, 50.0);
        let out = svg.render();
        println!("{out}");
        assert!(out.starts_with("<?xml"));
        assert!(out.contains("<svg xmlns=\"http://www.w3.org/2000/svg\""))
    }

    /// 各元素方法不 panic 且输出含对应标签
    #[test]
    fn test_svg_elements() {
        let mut svg = Svg::new(200.0, 100.0);
        svg.rect(1.0, 2.0, 3.0, 4.0, "#66c1f8", 0.38);
        svg.line(0.0, 0.0, 10.0, 10.0, "#999", 0.5);
        svg.polyline(&[(0.0, 0.0), (5.0, 5.0), (10.0, 2.0)], "#0e4fa0", 1.7, false);
        svg.circle(5.0, 5.0, 2.5, "#c44e52", 1.0);
        svg.text(10.0, 20.0, "期望评分", 12.0, "middle");
        let out = svg.render();
        println!("{out}");
        for tag in ["<rect", "<line", "<polyline", "<circle", "<text"] {
            assert!(out.contains(tag), "缺少 {tag}");
        }
        assert!(out.contains("期望评分"));
    }
}
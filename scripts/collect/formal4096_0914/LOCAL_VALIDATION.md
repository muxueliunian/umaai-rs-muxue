# 本机交付验证边界

- Python：旧驱动4项、新驱动4项测试通过；新清单100000含备用index校验通过。
- Rust：采集器与导出器Release构建通过；umasim全部bin Release check通过；
  umaai全部bin onnx Release check通过（曾发现EmptySink参数类型错误，修复后通过）。
- 手写normal/rollout整局动作与事件轨迹一致测试通过，评分均64918；不代表新旧教师比较。
- 新教师三年PT定价进入rollout的测试通过：无彩圈16、有彩圈36。
- 新友人rank合法性测试通过。
- 真R4模型、每候选4096的四类冒烟均完成并raw导出：
  通用1根2候选；地区各1根、10/10/120候选，turn=2/23/47。
  共142候选，零失败槽；CSR、slots/legal_mask、combo_fields/index映射通过。
  对raw以float64重算，均值与保存f32的最大绝对误差0.001953125，
  stdev最大绝对误差约0.00012123，符合f32舍入精度。
- 额外末次Release重链接不作为本机等待前置，云端必须从最终固定提交重新
  cargo build --release --locked并运行冒烟；不复用旧二进制。
- 未跑全量测试、Linux验证、云端号段检查或新旧教师强度对照；
  不执行内容指纹测试，不承诺新模型强度。
- 本机冒烟为合并待提交工作树构建，只作实现验证，不并入正式训练数据。
- 原master与原工作树修改保留，未启动正式采集。参考资产在隔离树
  target/newteacher4096_assets_lf，模型不进入Git。

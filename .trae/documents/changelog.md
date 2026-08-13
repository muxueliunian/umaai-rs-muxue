# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。

## 2026-08-13

### 文档整理
- 创建了AGENTS.md项目规则总结文档
- 在.trae/documents/目录下整理相关文档

### 测试规范完善
- 在umasim::utils中新增get_workspace_root()函数，用于获取workspace根目录
- 修改了多个测试文件，在测试中使用get_workspace_root()切换到workspace根目录

### 拉面剧本数据完善
- 更新ramen_basic_effect：添加jiban/status_limit/hint_special字段，填入3年效果数据
- 添加finals_effect：定义超级拉面(超RMJ極)的基础/额外/单独效果
- 添加ramen_region_effect：记录20条地域拉面效果数据
- 更新Rust结构体：添加RamenBasicEffect结构体
- 更新ramen_memo_cn.md文档：补充效果说明和字段文档

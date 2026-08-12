# UmaAI-RS 变更日志

本文件用于简要记录每次任务的修改内容。

## 2026-08-13

### 文档整理
- 创建了AGENTS.md项目规则总结文档
- 在.trae/documents/目录下整理相关文档

### 测试规范完善
- 在umasim::utils中新增get_workspace_root()函数，用于获取workspace根目录
- 修改了多个测试文件，在测试中使用get_workspace_root()切换到workspace根目录

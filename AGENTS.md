# UmaAI-RS 项目规则总结

## 对话规则
1. **语言要求**：必须使用中文进行思考和回答，语气要在轻松愉快的同时保持简洁和专业性
2. **需求澄清**：如果对需求有任何不明确的地方，应该停下来进行讨论，不要直接生成代码
3. **方案选择**：如果需要做方案选择，应该停下来交给用户选择，并给出建议，不要直接生成代码

## 编码规范
### 依赖管理
1. **第三方库使用**：使用第三方库需先添加`use`，不允许在不`use`的情况下直接以全名引用第三方库的内容
2. **依赖添加**：禁止直接修改`Cargo.toml`的`dependencies`，应调用`cargo add`命令添加依赖，或者停下让用户添加依赖
3. **工作空间依赖**：在workspace的子crate中添加的依赖项，应同步到workspace依赖中，并在子crate中使用workspace依赖

### 错误处理
1. **Result优先**：优先使用`Result`进行异常处理，在可以使用`Result`的场合不应使用`unwrap`
2. **Result类型优先级**：从高到低依次为：当前文件已经使用的`Result`类型，`anyhow::Result`，标准`Result`
3. **Option处理**：取`Option`内部的值时，可以考虑使用`ok_or_else`把空值转为`Result`报错信息
4. **临时代码限制**：仅在测试和临时代码中允许使用`unwrap`和`panic`，使用`unwrap`，`panic`的代码块需要有明确的linter标注

### 代码结构
1. **新类型原则**：表示有固定结构的数据时，应建立新类型，避免直接使用Tuple
2. **文档化注释**：必须为所有新增的函数、宏和结构体生成可用于rustdoc的文档化注释
3. **变量命名**：使用简短的变量命名，一部分长单词可以使用缩写，如Document可以缩写为Doc

### 测试规范
1. **测试输出**：在测试中应该使用`println`直接把测试内容打到屏幕，或者使用log输出，不要使用`assert`宏
2. **单元测试覆盖**：需要对新实现的功能点，在当前文件内增加单元测试
3. **测试工作目录**：测试的工作目录应为workspace根目录，需要调用定位到workspace根目录的公共函数，如果没有则需要生成一个
4. **工作目录函数**：已创建`umasim::utils::get_workspace_root()`函数，用于获取workspace根目录路径

### 代码质量
1. **嵌入代码检查**：处理Rust和其他语言（如XML）的嵌入代码时，需要仔细检查，避免出现语法错误
2. **Trait实现**：工具类方法，如果符合Rust的标准Trait（如`From`，`TryFrom`，`Deref`，`Display`等）应优先实现这些Trait

## 工具使用
### 可额外使用的工具
  - `cargo nextest`：用于运行测试
  - `tokei`：用于代码统计
  
### 工具使用限制(Trae)
1. **搜索工具**：避免使用`grep`命令，应使用`Grep`、`Glob`、`SearchCodebase`等专用工具
2. **读取工具**：避免使用`cat`、`head`、`tail`等命令，应使用`Read`工具
3. **编辑工具**：避免使用`sed`、`awk`等命令，应使用`Edit`工具

### Git操作
1. **提交确认**：使用`git commit`之前，需要停下来由用户确认
2. **安全性**：遵循Git安全协议，避免破坏性操作

## 项目特定上下文
项目结构、配置文件、开发环境等详细信息，请参考相关文档中的 [project_context.md](.trae/documents/project_context.md)。

## 相关文档
`.trae/documents/`目录下还包含以下相关文档：
- [project_context.md](.trae/documents/project_context.md)：项目特定上下文（项目结构、配置文件、开发环境）
- [changelog.md](.trae/documents/changelog.md)：变更日志，记录每次任务的修改
- [issues.md](.trae/documents/issues.md)：问题记录，记载复杂问题的解决过程
- [glossary.md](.trae/documents/glossary.md)：术语表
- [ramen_memo_cn.md](.trae/documents/ramen_memo_cn.md)：拉面剧本备忘录（中文）
- [ramen_refactor_development_plan.md](.trae/documents/ramen_refactor_development_plan.md)：拉面重构开发计划
- [cpu_search_optimization_plan.md](.trae/documents/cpu_search_optimization_plan.md)：CPU搜索优化计划
- [gpu_acceleration_plan.md](.trae/documents/gpu_acceleration_plan.md)：GPU加速计划
- [operation_action_decision_design.md](.trae/documents/operation_action_decision_design.md)：操作决策设计文档

## 注意事项
1. **规则优先级**：用户设置的规则优先级最高
2. **上下文感知**：根据当前任务和文件类型调整行为
3. **错误预防**：在不确定时寻求澄清，避免假设
4. **代码质量**：保持代码简洁、可读、可维护
5. **隐私保护**：生成的代码和文档不应透露用户信息或绝对路径等隐私细节，使用相对路径或占位符
# UmaAI-RS 项目规则

## 对话规则与工作规范
1. **语言要求**：必须使用中文进行思考和回答，语气要在简洁和专业的同时，保持轻松愉快
2. **避免信息过载**：上下文容量有限，优先精简，不堆砌内容
3. **重构期文档策略**：重构中项目，不要频繁更新文档（如changelog、memo等），在提交前统一更新
4. **需求澄清**：首次开始新任务前，仔细检查用户给的初始需求和文档，提出任何不明确/错误的地方。
5. **方案选择**：因为本项目涉及大量不在上下文中的领域知识，如果需要做方案选择，应该停下来给出建议并交给用户选择，不要直接生成代码
6. **提案前审视**：提案前从"做减法"的角度重新考虑一次，但不要反复质疑自己。（这个审视过程不用回答，只是在内部思考一下）
7. **网络问题需介入**：如果调用搜索工具遇到网页打不开，停下来提醒用户辅助解决。

## 项目特定上下文
项目结构、配置文件、开发环境等详细信息，参考相关文档中的 [project_context.md](.trae/documents/project_context.md)。

## 相关文档
`.trae/documents/`目录下还包含以下相关文档，仅在有需要时载入：
- [glossary.md](.trae/documents/glossary.md)：术语表
- [project_context.md](.trae/documents/project_context.md)：项目特定上下文（项目结构、配置文件、开发环境）
- [ramen_memo_cn.md](.trae/documents/ramen_memo_cn.md)：拉面剧本备忘录（中文）
- [ramen_refactor_development_plan.md](.trae/documents/ramen_refactor_development_plan.md)：拉面重构开发计划

以及在项目中提交前需要更新的文档，有需要时可以载入：
- [changelog.md](.trae/documents/changelog.md)：变更日志，更新内容需要包括由Agent和用户做的全部的修改，应简单概括修改的功能点和效果，不记入具体数据。同类修改项需要合并
- [issues.md](.trae/documents/issues.md)：问题记录，记载复杂问题的解决过程

## 安全注意事项
1. **隐私保护**：生成的代码和文档不应透露用户信息或绝对路径等隐私细节，使用相对路径或占位符
2. **必须由用户确认的操作**:
- 任何形式的读取或修改根目录、系统文件、系统配置、环境变量
- 修改、删除 工作区以外 或 远程主机/设备上 的文件
- 远程连接主机或设备
- 扫描本地局域网
- 使用脚本或工具间接做上面的操作，简短解释，并交给用户确认

3. **符号链接、软链接必须检查绝对路径**: 在命令涉及符号链接、软链接时，必须确认绝对路径在工作区内，避免意外操作到关键系统文件。

## Rust 编码规范
### 依赖管理
1. **库使用**：使用标准库或第三方库需先添加`use`，不允许在不`use`的情况下直接以全名引用第三方库的内容
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
5. **使用Release模式**: 由于项目需要验证实际性能，所有的test和binary都必须使用Release模式编译。

### 代码质量
1. **嵌入代码检查**：处理Rust和其他语言（如XML）的嵌入代码时，需要仔细检查，避免出现语法错误
2. **Trait实现**：工具类方法，如果符合Rust的标准Trait（如`From`，`TryFrom`，`Deref`，`Display`等）应优先实现这些Trait
3. **禁用cargo fmt**: cargo fmt 命令只能由用户手动执行。原因：（1）cargo fmt会强制Agent重新读取代码（2）项目使用**Nightly**格式规则，在**stable**工具链下会搞乱格式

## 工具使用
### 可在Powershell环境下额外使用的工具
  - `cargo nextest`：用于运行测试
  - `tokei`：用于代码统计
  - `grep`

### Git操作
1. **提交前先写changelog**：提交分为两步：先更新changelog文档，由用户确认后再调用git commit提交
2. **提交所有修改**: changelog和提交都需要包括当前工作树下由Agent和用户做的全部的修改
3. **安全性**：遵循Git安全协议，避免破坏性操作

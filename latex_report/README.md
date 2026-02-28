# LaTeX Report - Dino Jump with DQN

这是Dino Jump强化学习项目的完整LaTeX技术报告。

## 文件结构

```
latex_report/
├── main.tex              # 主LaTeX文档（40+ 页）
├── figures/              # 训练曲线图片文件夹
│   ├── v5.0_forgetting.png      # v5.0训练曲线（2000轮，展示遗忘）
│   ├── v6.1_clean_training.png  # v6.1 Clean成功案例
│   ├── v6.2_failed.png          # v6.2失败案例（-49%）
│   ├── v6.3-1_partial.png       # v6.3-1部分恢复
│   └── v6.3-2_tuning.png        # v6.3-2调参失败
├── compile.sh            # 一键编译脚本
└── README.md             # 本文档
```

## 快速开始

### 方法1: 使用编译脚本（推荐）

```bash
cd latex_report
./compile.sh
```

脚本会自动：
1. 检查LaTeX环境
2. 编译3遍（生成目录和交叉引用）
3. 清理辅助文件
4. 生成 `main.pdf`

### 方法2: 手动编译

```bash
cd latex_report
pdflatex main.tex
pdflatex main.tex  # 第二遍生成目录
pdflatex main.tex  # 第三遍确保引用正确
```

### 方法3: 使用LaTeX编辑器

推荐的编辑器：
- **Overleaf** (在线): 上传整个文件夹，自动编译
- **TeXShop** (macOS): 打开main.tex，点击"Typeset"
- **TeXworks** (跨平台): 打开main.tex，点击绿色按钮
- **VS Code** + LaTeX Workshop扩展

## 环境要求

### macOS
```bash
# 安装MacTeX (约4GB)
brew install --cask mactex

# 或安装BasicTeX (更小，约100MB)
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended
```

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

### Windows
下载并安装：
- **MiKTeX**: https://miktex.org/download
- 或 **TeX Live**: https://www.tug.org/texlive/

## 文档特点

### ✨ 内容亮点

1. **完整的版本历史** (v1.0 → v6.1 Clean)
   - 包括所有失败案例
   - 详细的错误分析
   - 系统性的恢复过程

2. **5张专业训练曲线图**
   - 每张图4个子图（分数、Loss、Epsilon、分布）
   - 高分辨率PNG图片
   - 详细的图注解释

3. **系统性消融研究**
   - Dropout: -16%
   - 课程学习: -30%
   - 奖励微调: ~0%
   - 组合: -49%

4. **专业LaTeX排版**
   - 精美的表格（booktabs）
   - 代码高亮（listings）
   - 彩色标注（成功/失败/警告）
   - 完整的交叉引用
   - 自动生成目录

### 📊 文档统计

- **页数**: 40+ 页
- **图表**: 5张训练曲线
- **表格**: 15+ 个专业表格
- **代码**: 8+ 个代码示例
- **章节**: 7个主要章节
- **参考文献**: 8篇

### 🎨 LaTeX特性

#### 颜色标注
- \textcolor{successgreen}{成功} - 绿色
- \textcolor{failurered}{失败} - 红色
- \textcolor{warningorange}{警告} - 橙色

#### 代码高亮
```latex
\begin{lstlisting}[language=Python, caption={示例}]
def forward(self, x):
    x = F.relu(self.fc1(x))
    return self.fc3(x)
\end{lstlisting}
```

#### 专业表格
- 使用 `booktabs` 包（\toprule, \midrule, \bottomrule）
- 使用 `tabularx` 实现自适应列宽
- 使用 `longtable` 支持跨页表格

#### 图片插入
```latex
\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/v6.1_clean_training.png}
\caption{训练曲线说明}
\label{fig:v6.1_clean}
\end{figure}
```

## 章节概览

### 1. Introduction (引言)
- 项目概述
- 技术栈介绍
- 项目演进统计

### 2. Game Design (游戏设计)
- 游戏规则和机制
- 类设计说明
- UI设计

### 3. Implementation (算法实现)
- Deep Q-Learning算法
- 网络架构
- 超参数配置
- 奖励函数

### 4. Version Evolution (版本演进) ⭐ **核心章节**
- v1.0-v4.0: 初期开发
- v5.0: 首次成功（图1）
- v6.0: 防遗忘机制
- v6.2: 失败优化（图2）
- v6.3-1: 部分恢复（图3）
- v6.3-2: 调参失败（图4）
- v6.1 Clean: 成功恢复（图5）
- 消融研究总结

### 5. Challenges and Solutions (挑战与解决)
- Dropout降级问题
- 课程学习失败
- 奖励函数设计
- "顿悟时刻"现象

### 6. Experimental Results (实验结果)
- 版本对比表
- 训练效率分析
- 最终模型性能

### 7. Conclusion (结论)
- 关键成就
- 核心教训
- 最终反思
- 领域贡献

### References (参考文献)
8篇学术论文和开源项目

## 自定义修改

### 修改图片
1. 替换 `figures/` 文件夹中的图片
2. 确保文件名匹配（或修改main.tex中的引用）
3. 推荐格式：PNG（高分辨率，300+ DPI）

### 修改内容
1. 打开 `main.tex`
2. 找到对应的章节
3. 编辑文本、表格或代码
4. 重新编译

### 添加新章节
```latex
\section{新章节标题}
\subsection{子章节}
内容...
```

### 添加新表格
```latex
\begin{table}[H]
\centering
\caption{表格标题}
\begin{tabular}{lcc}
\toprule
\textbf{列1} & \textbf{列2} & \textbf{列3} \\
\midrule
数据1 & 数据2 & 数据3 \\
\bottomrule
\end{tabular}
\end{table}
```

### 添加新图片
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.8\textwidth]{figures/new_image.png}
\caption{图片说明}
\label{fig:new_image}
\end{figure}
```

## 编译选项

### 完整编译（生成所有引用）
```bash
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

### 快速预览（跳过引用）
```bash
pdflatex main.tex
```

### 生成参考文献（如果使用BibTeX）
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### 清理辅助文件
```bash
rm -f main.aux main.log main.out main.toc main.bbl main.blg
```

## 常见问题

### Q1: 编译报错 "File `xxx.sty' not found"
**答**: 缺少LaTeX包，安装：
```bash
# macOS
sudo tlmgr install <package-name>

# Ubuntu
sudo apt-get install texlive-<package-name>
```

### Q2: 图片无法显示
**答**: 检查：
1. 图片文件是否在 `figures/` 文件夹
2. 文件名是否正确（区分大小写）
3. 图片格式是否支持（推荐PNG或PDF）

### Q3: 中文显示乱码
**答**: 本文档使用英文。如需中文支持，修改：
```latex
\usepackage[UTF8]{ctex}  % 添加到preamble
```

### Q4: 编译很慢
**答**: 正常现象。首次编译需要加载所有包和生成引用，需要1-2分钟。后续编译会更快。

### Q5: 参考文献未显示
**答**: 当前使用 `thebibliography` 环境（手动）。如需BibTeX：
1. 创建 `references.bib` 文件
2. 替换 `\begin{thebibliography}` 为 `\bibliography{references}`
3. 编译时运行 `bibtex main`

## 输出文件

编译成功后生成：
- **main.pdf** - 最终PDF文档（约2-3 MB）

辅助文件（可删除）：
- main.aux - 辅助信息
- main.log - 编译日志
- main.out - 超链接信息
- main.toc - 目录信息

## 高级功能

### 使用Overleaf在线编译
1. 访问 https://www.overleaf.com
2. 创建新项目（New Project → Upload Project）
3. 上传整个 `latex_report` 文件夹（压缩为zip）
4. 自动编译，实时预览

### 转换为其他格式

**转换为Word (.docx)**:
```bash
pandoc main.tex -o report.docx
```

**转换为HTML**:
```bash
pandoc main.tex -o report.html -s --mathjax
```

### 生成压缩PDF
```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
   -dNOPAUSE -dQUIET -dBATCH -sOutputFile=main_compressed.pdf main.pdf
```

## 打包分享

### 打包整个LaTeX项目
```bash
cd ..
tar -czf latex_report.tar.gz latex_report/
```

### 只打包必要文件
```bash
cd latex_report
zip -r report_latex.zip main.tex figures/ compile.sh README.md
```

## 技术支持

### 常用LaTeX资源
- **官方文档**: https://www.latex-project.org/
- **Overleaf教程**: https://www.overleaf.com/learn
- **TeX Stack Exchange**: https://tex.stackexchange.com/
- **CTAN包库**: https://ctan.org/

### 包文档查询
```bash
texdoc <package-name>
# 例如: texdoc booktabs
```

## 许可证

本LaTeX报告模板可自由使用和修改。

## 作者

- **Chao Wang**
- **项目**: Assignment1_RL_Game
- **日期**: February 2026

---

**祝编译顺利！如有问题请查看 main.log 文件。** 📄

# CEM-Mamba 代码说明

本仓库根据服务器上的双视图多任务实验代码整理，对应论文《A Dual-view Multi-task Deep Learning Framework for BI-RADS 4A Biopsy Reduction on Contrast-Enhanced Mammography》。完整安装、训练和评估命令见 [英文 README](README.md)。

模型包含共享 MambaVision 主干、CC/MLO 门控融合、良恶性分类及辅助 BI-RADS 连续风险回归；也保留了原代码的序数分类和比较模型选项。训练、单模型评估、交叉验证集成、阈值分析及 SmoothGrad 可视化均有独立入口。

**这份整理版尚不能称为论文结果的完整复现。** 实际检查到服务器当前导入的 selective scan 是直接返回输入的测试替身。正式整理版默认使用官方 `mamba-ssm`，同时将历史行为放在 `compat/legacy_scan.py`，仅通过环境变量 `CEM_SCAN_BACKEND=legacy_identity` 显式启用。切换实现会改变计算，不能把旧权重在新实现上的输出当作已验证结果。检查当前文件无法证明所有历史实验当时使用的实现。

## 使用顺序

1. 按英文 README 安装 PyTorch、依赖和官方 selective scan。
2. 在本地准备脱敏 CC/MLO ROI 图像，按 RE–LE–RE 顺序组成三通道输入；本仓库不包含 DICOM 转换与病灶标注流水线。
3. 参照 `examples/manifest.example.csv` 创建数据清单，保持患者级划分隔离。
4. 执行 `python scripts/validate_manifest.py data/manifest.csv` 检查清单。
5. 用 `scripts/train.py` 选择配置启动训练，再运行评估脚本。

`configs/manuscript_method.json` 是根据论文方法描述整理的示例，不是已核验的最终实验参数。`configs/observed_cv3_strongaug.json` 记录已查看的三折实验参数，原始初始化权重未包含。论文本身的折数描述不一致：研究设计写五折，训练与推理写三折；正文训练参数为 80 轮、辅助权重 0.2，已查看运行记录为 25 轮、辅助权重 0.1。详见 [复现差异](docs/REPRODUCIBILITY.md)。

## 上传 GitHub

将本目录内容作为仓库根目录上传即可。建议仓库名 `CEM-Mamba`。`.gitignore` 已排除患者数据、原始清单、权重、输出、日志、环境与密钥文件；示例 CSV 全部使用虚构标识符。不要把完整实验目录直接复制进来再上传。

README 中的 AUC 和活检减少率来自用户提供的论文，不是本次运行得出的结果。论文正文、患者影像、临床表格、训练权重均未随包发布，也未替你创建或推送远程仓库。

已保留第三方版权并补齐许可证。项目自有代码尚未指定公开使用许可证；没有擅自套用 MIT 或 Apache 许可。实际验证范围见 [验证记录](docs/VALIDATION.md)。

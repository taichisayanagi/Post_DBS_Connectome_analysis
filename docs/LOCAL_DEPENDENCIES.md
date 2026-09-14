# Local dependency policy

Approved 2026-09-15: use the researcher's existing software installations.
The distributed package contains our GUI, workflow code, configuration definitions and synthetic tests,
not an all-in-one neuroimaging environment.

## What is and is not distributed

| Item | Policy |
|---|---|
| This application's Python source and browser assets | Distributed under MIT |
| FSL, MRtrix3, FreeSurfer, MATLAB, SPM12 and other external tools | Use separate, user-managed local installations |
| NextBrain resources, atlases and other model weights | User-provided local resources under their original terms |
| Python interpreter and virtual environments | User-managed; not bundled |
| Python packages such as NumPy, SciPy, NiBabel and BrainSpace | Declared dependencies, not vendored; use a compatible dedicated environment |
| MATLAB Runtime and commercial toolboxes | Not bundled; requirements depend on the actual implemented processing route |
| Large container/VM images | Not required and not a planned release gate |
| Patient data or reconstructed anatomy | Never part of the package |

Installing this package does not install or configure the external neuroimaging suites.
Normal `pip install` may resolve/download the declared Python dependencies; this is different from
bundling those packages in our repository. Users who already have a verified compatible Python
environment can install with `python -m pip install --no-deps .`. That option does not check that
the environment is complete. Do not change a shared scientific environment without reviewing
the dependency changes; prefer an environment dedicated to this application.

## Required tools depend on the selected operation

Currently, DICOM inventory uses pydicom and conversion invokes dcm2niix. The new reconstruction
adapter calls MRtrix3, FSL eddy (optional topup) and ANTs. Prepared-input connectome execution invokes
MRtrix3, and gradient/QC operations use the Python dependencies declared in `pyproject.toml`.
FreeSurfer, NextBrain, MATLAB, SPM12 and other study tools will be connected as the corresponding
reconstruction stages are implemented and verified. Installation alone does **not** make those
currently missing stages operational. Gradient-only analysis should not require MATLAB or FreeSurfer.

Before implementing each upstream route, derive its exact dependency inventory from the study
scripts and execution records. Record required commands, MATLAB functions/toolboxes, models,
atlas resources and supported versions. Do not infer compatibility merely from a product name.

## Environment discovery and remaining checks

The GUI now offers read-only command discovery. It checks PATH, then explicitly configured
`FSLDIR`, `FREESURFER_HOME`, `MRTRIX_HOME` and `ANTSPATH` directories. It labels available commands
as **detected, not tested**. No program, MATLAB license checkout or package installation is triggered.
Paths/versions/hashes are additionally recorded when a processing command is explicitly executed.

The GUI will offer an environment-settings panel with explicit installation paths and a CLI
equivalent. Detection should be non-mutating: explicit paths first, then documented environment
variables and PATH. Do not silently install, update, downgrade or reconfigure external tools.
Do not assume that a GUI-launched process inherits the user's interactive shell environment.

Separate the following states:

- **Not found:** executable or required resource missing.
- **Detected, not tested:** a path exists, but an execution check has not passed.
- **Version unknown:** the version cannot be established reliably.
- **Version not validated:** executable available, but this application has no compatibility evidence.
- **Resource/license setup incomplete:** a required file/configuration is missing; report no license contents.
- **Smoke test passed:** the named, bounded test passed; this is not end-to-end or clinical validation.

MATLAB/SPM/toolbox execution tests must be explicit, because launching MATLAB may acquire a
license. Presence of a license file is not proof of entitlement or a usable license. Do not collect
license keys, read them into logs, or commit installation paths/configuration with the software.
Select and review CPU/GPU use before heavy processing; finding a GPU is not permission to use it.

## Reproducibility without bundling

Save executable paths, tool/package versions, applicable resource checksums, exact commands,
parameters, thread counts, inputs/outputs and QC attestations in the private run manifest.
Publish a tested compatibility matrix and installation instructions, not binaries or institutional
environment snapshots. Re-run frozen synthetic and study-level checks when dependency versions
change. Fail on a missing required dependency before beginning an expensive stage.

## 日本語の方針

既にPCにインストールされている解析ソフトウェアを呼び出す方式を正式方針とする．
本ソフトウェアはGUI，処理の順序制御，QC，設定と処理履歴の保存を担当する．
FSL，MRtrix3，FreeSurfer，NextBrain，MATLAB，SPM12，Python本体，atlas／重み等を一括同梱しない．
Docker等の大容量コンテナも必須にしない．

ただし「インストール済み」と「検証済みの組合せで実行可能」は異なる．
PATHと設定済みインストール先を読み取るGUIの検出機能を実装した．
追加資源，ライセンス設定，実行互換性の検査は今後の課題である．
使用する工程に不要な依存ソフトまで必須にせず，既存環境を断りなく変更しない．

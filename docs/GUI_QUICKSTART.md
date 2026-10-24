# GUI quick start — v0.3.0a3 research preview

## Install and launch

Use a dedicated Python 3.11+ environment. Install the lightweight package; install the required
MRI tools separately. See [dependency policy](LOCAL_DEPENDENCIES.md). macOS and Linux are the
intended environments; Windows/native MRI-tool interoperability has not been tested.

```sh
git clone https://github.com/taichisayanagi/Post_DBS_Connectome_analysis.git
cd Post_DBS_Connectome_analysis
git checkout v0.3.0a3
python -m pip install .
dbs-connectome gui --data-root /private/study_inputs --output-root /private/study_results
```

The input directory must already exist. The output directory must be separate from it and outside
the code checkout. Open the printed `127.0.0.1` URL on the same computer. Do not expose or port-forward
the server. Launch from a shell where the external tools' environment variables/PATH are set.
No patient information is uploaded. Nevertheless, output anatomy, metadata, logs and filenames
remain private information, including the self-contained HTML reports.

## Guided route

1. Run the read-only **installation check**. Detected does not mean functionally validated.
2. In **Image workflow**, select NIfTI inputs or inventory DICOM and explicitly assign series roles.
   Review patient/visit identity, b-values/directions, phase encoding, readout and reverse-PE inputs.
   No patient/series association is guessed from a filename.
3. Select CPU threads, review the plan, then authorize reconstruction. Inspect DWI correction,
   registered CT/T1 and b0. There is no automatic registration approval.
4. Open **CT / MRI electrode editor**. Generate CT shaft candidates with constrained MRI void, or
   manually seed shafts for MRI-only input. Correct points/local mask regions if needed, generate
   a new candidate, inspect every relevant slice and send the candidate to the main workflow.
5. Configure installed FreeSurfer/NextBrain/Schaefer resources and the frozen common node table.
   Run anatomical preparation. If correcting a completed preparation, use **Adopt corrected mask**;
   this creates a new preparation without rerunning unchanged anatomy or changing old results.
6. Open the mask/atlas/5TT review. A named reviewer must confirm all five checks before the final
   processing stage. Every changed mask invalidates earlier approval.
7. Run ACT tractography, exclusion, independent continuous-segment audit, SIFT2 refit and connectome.
   The gradient/matrix report appears under **QC / results**. A single-session embedding without
   a supplied reference is explicitly unaligned, not suitable for direct axis comparison.
8. Under **Training reference and longitudinal G1–G4**, supply masked-connectome run directories and
   the common node TSV. Build an explicitly reviewed training/independent mean reference, or use an
   existing reference. Review subject/visit pairing, then align both visits to that same reference.
   Open the change plot/table and `paired_gradient_change.csv`. Independent axis-stability and
   artifact-sensitivity assessments are still required. For cohort common/LOSO analysis,
   use the separate **Cohort / paper numerical replay** operation with a reviewed manifest.
9. Optional: configure **Electric field / VTA-like simulation** with a separately installed OSS-DBS
   environment. It is a different measurement from artifact exclusion. Read [all limitations](STIMULATION.md).

## Paper numerical replay

In **Advanced steps**, choose **Run reviewed numerical replay suite** and supply a hash-bound
suite JSON. The GUI reports progress for each cohort, sensitivity and control task and shows
passed/failed/unverified statuses. See [manifest definitions](PAPER_REPLAY.md) and
[the exact verification scope](VALIDATION_REPLAY_20261024.md).

To inspect the format without private patient data, run the synthetic example from the repository:

```sh
python examples/synthetic_study_demo.py --output-root /private/synthetic_demo
```

This writes artificial inputs and a suite configuration. Synthetic outputs without independent
expected values are deliberately **not** labelled as frozen-output verified. Do not add patient
inputs, clinical manifests, results or private HTML reports to a public repository.

The **Exact-test / detection-power audit** operation takes a `config` JSON such as
`examples/statistical_design.json`. It needs no patient images. Its browser report separates
exact n=6 ordering counts from Monte Carlo power estimates, records the random seed and
reports simulation uncertainty. It is not a power calculation based on validated clinical
effect sizes and does not certify the imaging pipeline.

The one-job queue shows elapsed time, stage and actual logs. `planned` is not completed processing;
`needs_qc` is not approval. On restart, load completed reconstruction/preparation directories explicitly.
An interrupted external process is not automatically resumed. Never edit completed outputs in place.

## 日本語の要点

解析操作はブラウザGUIで行える．初回の依存ツール設定，atlas資源の用意，共通nodeの定義，
画像登録の確認には専門家の関与が必要である．DICOMを入れるだけで全てを自動判断する製品ではない．
CTの金属候補とMRIの信号欠損を統合し，画像上で修正してからQC承認する設計である．
原画像は変更せず，修正後のマスクと解析結果は別ディレクトリへ保存する．
研究用プレビューであり，診断や刺激設定の決定には使用しない．

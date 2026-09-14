# Interface design

The local GUI follows the control-panel organization of established neuroimaging tools:
directory fields, grouped processing operations, explicit method settings, a run control,
and a persistent execution/log area. The mask viewer prioritizes the image planes.

The layout was informed by the [official Lead-DBS main-GUI screenshot](https://www.lead-dbs.org/helpsupport/knowledge-base/screenshots/)
(the linked historical v1.2 window, inspected 2026-09-15), not a claim of matching the latest Lead-DBS version.
No Lead-DBS source code, logos, screenshots or graphical assets are bundled or copied into this interface.
This is a separate application, not an official Lead-DBS component or endorsed extension.

## Working rules

- Use compact native controls and neutral grey panels; reserve color for selection and state.
- Keep input paths, fixed method parameters and execution conditions explicit.
- Display actual job states, elapsed times and logs; do not fabricate percentages or sample results.
- Retain research-only status and limitations without turning the workspace into a promotional page.
- Require the same input checks, mask-lineage validation and execution consent as the CLI.
- Keep patient-derived reports and computation local. Do not host this analysis server publicly.

This interface update changes presentation only. It does not implement the missing reconstruction
stages or establish end-to-end validity.
